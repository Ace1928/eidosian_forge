from unittest import TestCase
from traitlets import TraitError
from ipywidgets import FloatSlider
def test_construction_invalid_readout_format(self):
    with self.assertRaises(TraitError):
        FloatSlider(readout_format='broken')