from unittest import TestCase
from traitlets import TraitError
from ipywidgets import FloatSlider
def test_construction_readout_format(self):
    slider = FloatSlider(readout_format='$.1f')
    assert slider.get_state()['readout_format'] == '$.1f'