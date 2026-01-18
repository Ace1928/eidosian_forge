from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def make_stats_middleware(self, stat_name=None, stats_host=None, remove_uuid=False, remove_short_uuid=False):
    if stat_name is None:
        stat_name = uuid.uuid4().hex
    if stats_host is None:
        stats_host = uuid.uuid4().hex
    conf = dict(name=stat_name, stats_host=stats_host, remove_uuid=remove_uuid, remove_short_uuid=remove_short_uuid)

    @webob.dec.wsgify
    def fake_application(req):
        return 'Hello, World'
    return stats.StatsMiddleware(fake_application, conf)