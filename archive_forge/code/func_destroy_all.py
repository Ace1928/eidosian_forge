import atexit
from collections import OrderedDict
@classmethod
def destroy_all(cls):
    """Destroy all figures."""
    for manager in list(cls.figs.values()):
        manager.canvas.mpl_disconnect(manager._cidgcf)
        manager.destroy()
    cls.figs.clear()