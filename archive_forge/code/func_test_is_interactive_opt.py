import holoviews
import pytest
from holoviews.core import Store
from holoviews.element import Area, Curve
from hvplot.backend_transforms import (
@pytest.mark.parametrize(('bk_option', 'expected'), (('height', False), ('hover_line_alpha', True), ('nonselection_line_alpha', True), ('muted_line_alpha', True), ('selection_line_alpha', True), ('annular_muted_alpha', True)))
def test_is_interactive_opt(bk_option, expected):
    assert _is_interactive_opt(bk_option) == expected