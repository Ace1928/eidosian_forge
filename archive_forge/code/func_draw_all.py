import atexit
from collections import OrderedDict
@classmethod
def draw_all(cls, force=False):
    """
        Redraw all stale managed figures, or, if *force* is True, all managed
        figures.
        """
    for manager in cls.get_all_fig_managers():
        if force or manager.canvas.figure.stale:
            manager.canvas.draw_idle()