import functools
import time
import uuid
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import uuidutils
from six.moves import range  # noqa
from os_win._i18n import _
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
from os_win.utils import pathutils
def get_vm_power_state_change_listener(self, timeframe=_DEFAULT_EVENT_CHECK_TIMEFRAME, event_timeout=constants.DEFAULT_WMI_EVENT_TIMEOUT_MS, filtered_states=None, get_handler=False):
    field = self._VM_ENABLED_STATE_PROP
    query = self._get_event_wql_query(cls=self._COMPUTER_SYSTEM_CLASS, field=field, timeframe=timeframe, filtered_states=filtered_states)
    listener = self._conn.Msvm_ComputerSystem.watch_for(raw_wql=query, fields=[field])

    def _handle_events(callback):
        if patcher.is_monkey_patched('thread'):
            listen = functools.partial(tpool.execute, listener, event_timeout)
        else:
            listen = functools.partial(listener, event_timeout)
        while True:
            try:
                event = listen()
                vm_name = event.ElementName
                vm_state = event.EnabledState
                vm_power_state = self.get_vm_power_state(vm_state)
                try:
                    callback(vm_name, vm_power_state)
                except Exception:
                    err_msg = 'Executing VM power state change event callback failed. VM name: %(vm_name)s, VM power state: %(vm_power_state)s.'
                    LOG.exception(err_msg, dict(vm_name=vm_name, vm_power_state=vm_power_state))
            except exceptions.x_wmi_timed_out:
                pass
            except Exception:
                LOG.exception('The VM power state change event listener encountered an unexpected exception.')
                time.sleep(event_timeout / 1000)
    return _handle_events if get_handler else listener