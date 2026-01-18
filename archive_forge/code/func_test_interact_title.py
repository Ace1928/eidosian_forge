from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_interact_title():

    def test(a):
        return a
    interact_pane = interactive(test, a=False, params={'name': 'Test'})
    html = interact_pane.widget_box[0]
    assert isinstance(html, HTML)
    assert html.object == '<h2>Test</h2>'