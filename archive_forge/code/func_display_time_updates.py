import inspect
import os
import threading
import time
import warnings
from modin.config import Engine, ProgressBar
def display_time_updates(bar):
    """
    Start displaying the progress `bar` in a notebook.

    Parameters
    ----------
    bar : tqdm.tqdm
        The progress bar wrapper to display in a notebook cell.
    """
    threading.Thread(target=_show_time_updates, args=(bar,)).start()