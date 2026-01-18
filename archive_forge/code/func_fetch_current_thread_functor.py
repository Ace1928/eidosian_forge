import threading
import warnings
from oslo_utils import importutils
from oslo_utils import timeutils
def fetch_current_thread_functor():
    """Get the current thread.

    If eventlet is used to monkey-patch the threading module, return the
    current eventlet greenthread. Otherwise, return the current Python thread.

    .. versionadded:: 1.5
    """
    if not EVENTLET_AVAILABLE:
        return threading.current_thread
    else:
        green_threaded = _patcher.is_monkey_patched('thread')
        if green_threaded:
            return _eventlet.getcurrent
        else:
            return threading.current_thread