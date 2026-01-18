import pytest
from matplotlib.font_manager import FontProperties
def test_fontconfig_unknown_constant():
    with pytest.warns(DeprecationWarning):
        FontProperties(':unknown')