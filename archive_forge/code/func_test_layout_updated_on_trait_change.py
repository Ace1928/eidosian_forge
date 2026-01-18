from unittest import TestCase
from unittest import mock
import pytest
import traitlets
import ipywidgets as widgets
from ipywidgets.widgets.widget_templates import LayoutProperties
def test_layout_updated_on_trait_change(self):
    """test whether respective layout traits are updated when traits change"""
    template = self.DummyTemplate(width='100%')
    assert template.width == '100%'
    assert template.layout.width == '100%'
    template.width = 'auto'
    assert template.width == 'auto'
    assert template.layout.width == 'auto'