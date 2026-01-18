import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_Viewer_not_initialized():

    class Test(Viewer):

        def __panel__(self):
            return '# Test'
    test = panel(Test)
    assert test.object == '# Test'
    test = panel(Test())
    assert test.object == '# Test'