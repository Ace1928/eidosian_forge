from bokeh.events import ButtonClick, MenuItemClick
from panel.widgets import Button, MenuButton, Toggle
def test_button_event(document, comm):
    button = Button(name='Button')
    widget = button.get_root(document, comm=comm)
    events = []

    def callback(event):
        events.append(event.new)
    button.param.watch(callback, 'value')
    assert button.value == False
    button._process_event(ButtonClick(widget))
    assert events == [True]
    assert button.value == False