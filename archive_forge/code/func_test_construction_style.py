from unittest import TestCase
from traitlets import TraitError
import ipywidgets as widgets
def test_construction_style(self):
    box = widgets.Box(box_style='warning')
    assert box.get_state()['box_style'] == 'warning'