from unittest import TestCase
from traitlets import TraitError
import ipywidgets as widgets
def test_construction_invalid_style(self):
    with self.assertRaises(TraitError):
        widgets.Box(box_style='invalid')