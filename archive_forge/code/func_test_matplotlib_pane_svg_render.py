from bokeh.models import Div, Row as BkRow
import panel as pn
from panel.pane import Bokeh, Matplotlib, PaneBase
from panel.tests.util import mpl_available, mpl_figure
@mpl_available
def test_matplotlib_pane_svg_render(document, comm):
    pane = pn.pane.Matplotlib(mpl_figure(), format='svg', encode=True)
    model = pane.get_root(document, comm=comm)
    assert model.text.startswith('&lt;img src=&quot;data:image/svg+xml;base64,')