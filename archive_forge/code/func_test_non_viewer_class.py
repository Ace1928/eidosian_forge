import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_non_viewer_class():

    class Example:

        def __panel__(self):
            return 42
    panel(Example())