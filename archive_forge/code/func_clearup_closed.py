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
def clearup_closed(self):
    """Clear up any closed Comms."""
    self.web_sockets = {socket for socket in self.web_sockets if socket.is_open()}
    if len(self.web_sockets) == 0:
        CloseEvent('close_event', self.canvas)._process()