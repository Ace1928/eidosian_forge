import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
def test_format_using_callable():

    def show_three_decimal_places(value):
        return f'{value:.3f}'
    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1], format=show_three_decimal_places)
    assert s.diagrams[0].texts[0].get_text() == 'First\n0.250'