import time
from contextlib import contextmanager
from functools import partial
from ssl import CERT_NONE, CERT_OPTIONAL, CERT_REQUIRED
from urllib.parse import unquote
from kombu.utils.functional import retry_over_time
from kombu.utils.objects import cached_property
from kombu.utils.url import _parse_url, maybe_sanitize_url
from celery import states
from celery._state import task_join_will_block
from celery.canvas import maybe_signature
from celery.exceptions import BackendStoreError, ChordError, ImproperlyConfigured
from celery.result import GroupResult, allow_join_result
from celery.utils.functional import _regen, dictfilter
from celery.utils.log import get_logger
from celery.utils.time import humanize_seconds
from .asynchronous import AsyncBackendMixin, BaseResultConsumer
from .base import BaseKeyValueStoreBackend
def on_chord_part_return(self, request, state, result, propagate=None, **kwargs):
    app = self.app
    tid, gid, group_index = (request.id, request.group, request.group_index)
    if not gid or not tid:
        return
    if group_index is None:
        group_index = '+inf'
    client = self.client
    jkey = self.get_key_for_group(gid, '.j')
    tkey = self.get_key_for_group(gid, '.t')
    skey = self.get_key_for_group(gid, '.s')
    result = self.encode_result(result, state)
    encoded = self.encode([1, tid, state, result])
    with client.pipeline() as pipe:
        pipeline = (pipe.zadd(jkey, {encoded: group_index}).zcount(jkey, '-inf', '+inf') if self._chord_zset else pipe.rpush(jkey, encoded).llen(jkey)).get(tkey).get(skey)
        if self.expires:
            pipeline = pipeline.expire(jkey, self.expires).expire(tkey, self.expires).expire(skey, self.expires)
        _, readycount, totaldiff, chord_size_bytes = pipeline.execute()[:4]
    totaldiff = int(totaldiff or 0)
    if chord_size_bytes:
        try:
            callback = maybe_signature(request.chord, app=app)
            total = int(chord_size_bytes) + totaldiff
            if readycount == total:
                header_result = GroupResult.restore(gid)
                if header_result is not None:
                    header_result.on_ready()
                    join_func = header_result.join_native if header_result.supports_native_join else header_result.join
                    with allow_join_result():
                        resl = join_func(timeout=app.conf.result_chord_join_timeout, propagate=True)
                else:
                    decode, unpack = (self.decode, self._unpack_chord_result)
                    with client.pipeline() as pipe:
                        if self._chord_zset:
                            pipeline = pipe.zrange(jkey, 0, -1)
                        else:
                            pipeline = pipe.lrange(jkey, 0, total)
                        resl, = pipeline.execute()
                    resl = [unpack(tup, decode) for tup in resl]
                try:
                    callback.delay(resl)
                except Exception as exc:
                    logger.exception('Chord callback for %r raised: %r', request.group, exc)
                    return self.chord_error_from_stack(callback, ChordError(f'Callback error: {exc!r}'))
                finally:
                    with client.pipeline() as pipe:
                        pipe.delete(jkey).delete(tkey).delete(skey).execute()
        except ChordError as exc:
            logger.exception('Chord %r raised: %r', request.group, exc)
            return self.chord_error_from_stack(callback, exc)
        except Exception as exc:
            logger.exception('Chord %r raised: %r', request.group, exc)
            return self.chord_error_from_stack(callback, ChordError(f'Join error: {exc!r}'))