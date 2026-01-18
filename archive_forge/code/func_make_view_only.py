import codecs
import io
from .. import tests
from ..progress import ProgressTask
from ..ui.text import TextProgressView
from . import ui_testing
def make_view_only(self, out, width=79):
    view = TextProgressView(out)
    view._avail_width = lambda: width
    return view