from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_discrete_slider(document, comm):
    discrete_slider = DiscreteSlider(name='DiscreteSlider', value=1, options=[0.1, 1, 10, 100])
    box = discrete_slider.get_root(document, comm=comm)
    label = box.children[0]
    widget = box.children[1]
    assert isinstance(label, BkDiv)
    assert isinstance(widget, BkSlider)
    assert widget.value == 1
    assert widget.start == 0
    assert widget.end == 3
    assert widget.step == 1
    assert label.text == 'DiscreteSlider: <b>1</b>'
    discrete_slider._slider._process_events({'value': 2})
    assert discrete_slider.value == 10
    discrete_slider._slider._process_events({'value_throttled': 2})
    assert discrete_slider.value_throttled == 10
    discrete_slider.value = 100
    assert widget.value == 3
    with config.set(throttled=True):
        discrete_slider._slider._process_events({'value': 0.1})
        assert discrete_slider.value == 100
        discrete_slider._slider._process_events({'value_throttled': 0.1})
        assert discrete_slider.value == 0.1
        discrete_slider.value = 1
        assert widget.value == 1