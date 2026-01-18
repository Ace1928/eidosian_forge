from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_integer_interact():

    def test(a):
        return a
    interact_pane = interactive(test, a=1)
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.IntSlider)
    assert widget.value == 1
    assert widget.start == -1
    assert widget.step == 1
    assert widget.end == 3