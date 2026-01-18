import pytest
from pyparsing import ParseException
from ..parsing import (
from ..testing import requires
@pytest.mark.parametrize('species, html', [('Li@C60', 'Li@C<sub>60</sub>'), ('(Li@C60)+', '(Li@C<sub>60</sub>)<sup>+</sup>'), ('Na@C60', 'Na@C<sub>60</sub>'), ('(Na@C60)+', '(Na@C<sub>60</sub>)<sup>+</sup>')])
@requires(parsing_library)
def test_formula_to_html_caged(species, html):
    """Should produce HTML for cage species."""
    assert formula_to_html(species) == html