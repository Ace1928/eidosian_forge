import pytest
from pandas import Series
from pandas.plotting._matplotlib.style import get_standard_colors
@pytest.mark.parametrize('color', ['bad_color', ('red', 'green', 'bad_color'), (0.1,), (0.1, 0.2), (0.1, 0.2, 0.3, 0.4, 0.5)])
def test_bad_color_raises(self, color):
    with pytest.raises(ValueError, match='Invalid color'):
        get_standard_colors(color=color, num_colors=5)