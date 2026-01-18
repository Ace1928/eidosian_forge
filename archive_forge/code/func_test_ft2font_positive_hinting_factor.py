from pathlib import Path
import io
import pytest
from matplotlib import ft2font
from matplotlib.testing.decorators import check_figures_equal
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
def test_ft2font_positive_hinting_factor():
    file_name = fm.findfont('DejaVu Sans')
    with pytest.raises(ValueError, match='hinting_factor must be greater than 0'):
        ft2font.FT2Font(file_name, 0)