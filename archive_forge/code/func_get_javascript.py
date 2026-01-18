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
def get_javascript(cls, stream=None):
    if stream is None:
        output = io.StringIO()
    else:
        output = stream
    super().get_javascript(stream=output)
    output.write((pathlib.Path(__file__).parent / 'web_backend/js/nbagg_mpl.js').read_text(encoding='utf-8'))
    if stream is None:
        return output.getvalue()