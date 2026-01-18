from base64 import b64encode
import io
import json
import pathlib
import uuid
from ipykernel.comm import Comm
from IPython.display import display, Javascript, HTML
from matplotlib import is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2
from .backend_webagg_core import (
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611
@classmethod
def create_with_canvas(cls, canvas_class, figure, num):
    canvas = canvas_class(figure)
    manager = cls(canvas, num)
    if is_interactive():
        manager.show()
        canvas.draw_idle()

    def destroy(event):
        canvas.mpl_disconnect(cid)
        Gcf.destroy(manager)
    cid = canvas.mpl_connect('close_event', destroy)
    return manager