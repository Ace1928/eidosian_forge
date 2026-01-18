import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_bound_function_to_kwarg():
    widget = IntSlider(value=1)

    def add1(value):
        return value + 1

    def divide(divisor=2, value=0):
        return value / divisor
    bound_function = bind(divide, value=bind(add1, widget.param.value))
    assert bound_function() == 1
    widget.value = 3
    assert bound_function() == 2