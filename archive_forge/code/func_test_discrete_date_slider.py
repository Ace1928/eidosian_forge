from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_discrete_date_slider(document, comm):
    dates = {'2016-01-0%d' % i: datetime(2016, 1, i) for i in range(1, 4)}
    discrete_slider = DiscreteSlider(name='DiscreteSlider', value=dates['2016-01-02'], options=dates)
    box = discrete_slider.get_root(document, comm=comm)
    assert isinstance(box, BkColumn)
    label = box.children[0]
    widget = box.children[1]
    assert isinstance(label, BkDiv)
    assert isinstance(widget, BkSlider)
    assert widget.value == 1
    assert widget.start == 0
    assert widget.end == 2
    assert widget.step == 1
    assert label.text == 'DiscreteSlider: <b>2016-01-02</b>'
    discrete_slider._slider._process_events({'value': 2})
    assert discrete_slider.value == dates['2016-01-03']
    discrete_slider._slider._process_events({'value_throttled': 2})
    assert discrete_slider.value_throttled == dates['2016-01-03']
    discrete_slider.value = dates['2016-01-01']
    assert widget.value == 0
    with config.set(throttled=True):
        discrete_slider._slider._process_events({'value': 2})
        assert discrete_slider.value == dates['2016-01-01']
        discrete_slider._slider._process_events({'value_throttled': 2})
        assert discrete_slider.value == dates['2016-01-03']
        discrete_slider.value = dates['2016-01-02']
        assert widget.value == 1