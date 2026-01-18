import inspect
import os
import threading
import time
import warnings
from modin.config import Engine, ProgressBar
def progress_bar_wrapper(f):
    """
    Wrap computation function inside a progress bar.

    Spawns another thread which displays a progress bar showing
    estimated completion time.

    Parameters
    ----------
    f : callable
        The name of the function to be wrapped.

    Returns
    -------
    callable
        Decorated version of `f` which reports progress.
    """
    from functools import wraps

    @wraps(f)
    def magic(*args, **kwargs):
        result_parts = f(*args, **kwargs)
        if ProgressBar.get():
            current_frame = inspect.currentframe()
            function_name = None
            while function_name != '<module>':
                filename, line_number, function_name, lines, index = inspect.getframeinfo(current_frame)
                current_frame = current_frame.f_back
            t = threading.Thread(target=call_progress_bar, args=(result_parts, line_number))
            t.start()
            from IPython import get_ipython
            try:
                ipy_str = str(type(get_ipython()))
                if 'zmqshell' not in ipy_str:
                    t.join()
            except Exception:
                pass
        return result_parts
    return magic