import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('num_colors, expected_name', [(1, ['C0']), (3, ['C0', 'C1', 'C2']), (12, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0', 'C1'])])
def test_default_colors_named_undefined_prop_cycle(self, num_colors, expected_name):
    import matplotlib as mpl
    import matplotlib.colors as mcolors
    with mpl.rc_context(rc={}):
        expected = [mcolors.to_hex(x) for x in expected_name]
        result = get_standard_colors(num_colors=num_colors)
        assert result == expected