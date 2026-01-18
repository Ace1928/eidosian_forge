import pydevd_tracing
import greenlet
import gevent
from _pydev_bundle._pydev_saved_modules import threading
from _pydevd_bundle.pydevd_custom_frames import add_custom_frame, update_custom_frame, remove_custom_frame
from _pydevd_bundle.pydevd_constants import GEVENT_SHOW_PAUSED_GREENLETS, get_global_debugger, \
from _pydev_bundle import pydev_log
from pydevd_file_utils import basename
def greenlet_events(event, args):
    pydevd_tracing.reapply_settrace()