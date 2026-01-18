import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('num_colors, expected', [(3, ['red', 'green', 'blue']), (5, ['red', 'green', 'blue', 'red', 'green']), (7, ['red', 'green', 'blue', 'red', 'green', 'blue', 'red']), (2, ['red', 'green']), (1, ['red'])])
def test_default_colors_named_from_prop_cycle(self, num_colors, expected):
    import matplotlib as mpl
    from matplotlib.pyplot import cycler
    mpl_params = {'axes.prop_cycle': cycler(color=['red', 'green', 'blue'])}
    with mpl.rc_context(rc=mpl_params):
        result = get_standard_colors(num_colors=num_colors)
        assert result == expected