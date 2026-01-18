from datetime import date, datetime
import pytest
from bokeh.models import (
from panel import config
from panel.widgets import (
def test_discrete_slider_disabled(document, comm):
    discrete_slider = DiscreteSlider(name='DiscreteSlider', options=[0, 1], disabled=True)
    box = discrete_slider.get_root(document, comm=comm)
    assert box.children[0].text == 'DiscreteSlider: <b>0</b>'
    assert box.children[1].disabled
    assert box.children[1].start == 0
    assert box.children[1].end == 1
    discrete_slider.disabled = False
    assert box.children[0].text == 'DiscreteSlider: <b>0</b>'
    assert not box.children[1].disabled
    assert box.children[1].start == 0
    assert box.children[1].end == 1
    discrete_slider.options = [0]
    discrete_slider.disabled = True
    assert box.children[0].text == 'DiscreteSlider: <b>0</b>'
    assert box.children[1].disabled
    assert box.children[1].start == 0
    assert box.children[1].end == 1