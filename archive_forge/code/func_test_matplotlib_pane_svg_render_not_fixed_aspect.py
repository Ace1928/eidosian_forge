from bokeh.models import Div, Row as BkRow
import panel as pn
from panel.pane import Bokeh, Matplotlib, PaneBase
from panel.tests.util import mpl_available, mpl_figure
@mpl_available
def test_matplotlib_pane_svg_render_not_fixed_aspect(document, comm):
    pane = pn.pane.Matplotlib(mpl_figure(), format='svg', encode=False, fixed_aspect=False)
    model = pane.get_root(document, comm=comm)
    assert model.text.count('width=&quot;100%&quot;') == 1
    assert model.text.count('height=&quot;100%&quot;') == 1
    assert model.text.count('preserveAspectRatio=&quot;none&quot;') == 1