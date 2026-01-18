from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_align_items_extra_options(self):
    template = self.DummyTemplate(align_items='top')
    assert template.align_items == 'top'
    assert template.layout.align_items == 'flex-start'
    template.align_items = 'bottom'
    assert template.align_items == 'bottom'
    assert template.layout.align_items == 'flex-end'