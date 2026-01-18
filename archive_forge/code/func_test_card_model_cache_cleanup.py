from bokeh.models import Div
from panel.layout import Card
from panel.models import Card as CardModel
from panel.pane import HTML
def test_card_model_cache_cleanup(document, comm):
    html = HTML()
    l = Card(header=html)
    model = l.get_root(document, comm)
    ref = model.ref['id']
    assert ref in l._models
    assert l._models[ref] == (model, None)
    assert ref in html._models
    l._cleanup(model)
    assert l._models == {}
    assert html._models == {}