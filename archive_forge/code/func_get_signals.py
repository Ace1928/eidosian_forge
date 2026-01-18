from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from urllib import parse
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients.os import swift
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine import support
def get_signals(self):
    try:
        container = self.client().get_container(self.stack.id)
    except Exception as exc:
        self.client_plugin().ignore_not_found(exc)
        LOG.debug('Swift container %s was not found', self.stack.id)
        return []
    index = container[1]
    if not index:
        LOG.debug('Swift objects in container %s were not found', self.stack.id)
        return []
    filtered = [obj for obj in index if self.obj_name in obj['name']]
    obj_bodies = []
    for obj in filtered:
        try:
            signal = self.client().get_object(self.stack.id, obj['name'])
        except Exception as exc:
            self.client_plugin().ignore_not_found(exc)
            continue
        body = signal[1]
        if isinstance(body, bytes):
            body = body.decode()
        if body == swift.IN_PROGRESS:
            continue
        if body == '':
            obj_bodies.append({})
            continue
        try:
            obj_bodies.append(jsonutils.loads(body))
        except ValueError:
            raise exception.Error(_('Failed to parse JSON data: %s') % body)
    signals = []
    signal_num = 1
    for signal in obj_bodies:
        sig_id = self.UNIQUE_ID
        ids = [s.get(sig_id) for s in signals if sig_id in s]
        if ids and sig_id in signal and (ids.count(signal[sig_id]) > 0):
            [signals.remove(s) for s in signals if s.get(sig_id) == signal[sig_id]]
        signal.setdefault(self.DATA, None)
        unique_id = signal.setdefault(sig_id, signal_num)
        reason = 'Signal %s received' % unique_id
        signal.setdefault(self.REASON, reason)
        signal.setdefault(self.STATUS, self.STATUS_SUCCESS)
        signals.append(signal)
        signal_num += 1
    return signals