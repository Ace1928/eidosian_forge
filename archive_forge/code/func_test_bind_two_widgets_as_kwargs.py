import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_two_widgets_as_kwargs():
    widget = IntSlider(value=0)
    widget2 = IntSlider(value=1)

    def add(value, value2):
        return value + value2
    bound_function = bind(add, value=widget, value2=widget2)
    assert bound_function() == 1
    widget.value = 1
    assert bound_function() == 2
    widget2.value = 2
    assert bound_function() == 3
    with pytest.raises(TypeError):
        bound_function(1, 2)
    assert bound_function(value2=5) == 6