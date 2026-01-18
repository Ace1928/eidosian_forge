import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_bound_function_to_arg():
    widget = IntSlider(value=1)

    def add1(value):
        return value + 1

    def divide(value):
        return value / 2
    bound_function = bind(divide, bind(add1, widget.param.value))
    assert bound_function() == 1
    widget.value = 3
    assert bound_function() == 2