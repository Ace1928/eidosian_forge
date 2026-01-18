from bokeh.models import Div
from panel.layout import Card
from panel.models import Card as CardModel
from panel.pane import HTML
def test_card_get_root(document, comm):
    div1 = Div()
    div2 = Div()
    layout = Card(div1, div2)
    model = layout.get_root(document, comm=comm)
    ref = model.ref['id']
    header = layout._header_layout._models[ref][0]
    assert isinstance(model, CardModel)
    assert model.children == [header, div1, div2]
    assert header.children[0].text == '&amp;#8203;'