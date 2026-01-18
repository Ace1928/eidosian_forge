import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('color', ['', [], (), Series([], dtype='object')])
def test_empty_color_raises(self, color):
    with pytest.raises(ValueError, match='Invalid color argument'):
        get_standard_colors(color=color, num_colors=1)