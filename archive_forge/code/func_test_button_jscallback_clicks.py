from bokeh.events import ButtonClick, MenuItemClick
from panel.widgets import Button, MenuButton, Toggle
def test_button_jscallback_clicks(document, comm):
    button = Button(name='Button')
    code = 'console.log("Clicked!")'
    button.jscallback(clicks=code)
    widget = button.get_root(document, comm=comm)
    assert len(widget.js_event_callbacks) == 1
    callbacks = widget.js_event_callbacks
    assert 'button_click' in callbacks
    assert len(callbacks['button_click']) == 1
    assert code in callbacks['button_click'][0].code