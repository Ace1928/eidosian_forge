import warnings
import pytest
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.xfail(raises=RuntimeError, strict=True, reason='Test for check_figures_equal test creating new figures')
@check_figures_equal()
def test_check_figures_equal_extra_fig(fig_test, fig_ref):
    plt.figure()