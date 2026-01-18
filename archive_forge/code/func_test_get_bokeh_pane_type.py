from bokeh.models import Div, Row as BkRow
import panel as pn
from panel.pane import Bokeh, Matplotlib, PaneBase
from panel.tests.util import mpl_available, mpl_figure
def test_get_bokeh_pane_type():
    div = Div()
    assert PaneBase.get_pane_type(div) is Bokeh