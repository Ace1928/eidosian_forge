from _pydevd_bundle.pydevd_constants import (STATE_RUN, PYTHON_SUSPEND, SUPPORT_GEVENT, ForkSafeLock,
from _pydev_bundle import pydev_log
from _pydev_bundle._pydev_saved_modules import threading
import weakref
def any_thread_stepping():
    return bool(_infos_stepping)