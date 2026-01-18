import param
import pytest
from panel import config
from panel.interact import interactive
from panel.pane import Markdown, Str, panel
from panel.param import ParamMethod
from panel.viewable import Viewable, Viewer
from .util import jb_available
def test_clone_with_non_defaults():
    v = Viewable(loading=True)
    clone = v.clone()
    assert [(k, v) for k, v in sorted(v.param.values().items()) if k not in 'name'] == [(k, v) for k, v in sorted(clone.param.values().items()) if k not in 'name']