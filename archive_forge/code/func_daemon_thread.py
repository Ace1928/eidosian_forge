import _thread
import collections
import multiprocessing
import threading
from taskflow.utils import misc
def daemon_thread(target, *args, **kwargs):
    """Makes a daemon thread that calls the given target when started."""
    thread = threading.Thread(target=target, args=args, kwargs=kwargs)
    thread.daemon = True
    return thread