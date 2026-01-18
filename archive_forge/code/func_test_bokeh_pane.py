from bokeh.models import Div, Row as BkRow
import panel as pn
from panel.pane import Bokeh, Matplotlib, PaneBase
from panel.tests.util import mpl_available, mpl_figure
def test_bokeh_pane(document, comm):
    div = Div()
    pane = pn.panel(div)
    row = pane.get_root(document, comm=comm)
    assert isinstance(row, BkRow)
    assert len(row.children) == 1
    model = row.children[0]
    assert model is div
    assert pane._models[row.ref['id']][0] is model
    div2 = Div()
    pane.object = div2
    new_model = row.children[0]
    assert new_model is div2
    assert pane._models[row.ref['id']][0] is new_model
    pane._cleanup(row)
    assert pane._models == {}