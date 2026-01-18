from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_discrete_slider_label_update(document, comm):
    discrete_slider = DiscreteSlider(name='DiscreteSlider', value=1, options=[0.1, 1, 10, 100])
    box = discrete_slider.get_root(document, comm=comm)
    discrete_slider.value = 100
    assert box.children[0].text == 'DiscreteSlider: <b>100</b>'