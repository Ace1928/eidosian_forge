import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['fancyarrow_dpi_cor_200dpi.png'], remove_text=True, tol=0 if platform.machine() == 'x86_64' else 0.02, savefig_kwarg=dict(dpi=200))
def test_fancyarrow_dpi_cor_200dpi():
    """
    As test_fancyarrow_dpi_cor_100dpi, but exports @ 200 DPI. The relative size
    of the arrow head should be the same.
    """
    __prepare_fancyarrow_dpi_cor_test()