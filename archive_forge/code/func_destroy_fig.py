import atexit
from collections import OrderedDict
@classmethod
def destroy_fig(cls, fig):
    """Destroy figure *fig*."""
    num = next((manager.num for manager in cls.figs.values() if manager.canvas.figure == fig), None)
    if num is not None:
        cls.destroy(num)