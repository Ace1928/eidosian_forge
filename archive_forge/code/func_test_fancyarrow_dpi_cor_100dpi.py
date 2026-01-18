import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['fancyarrow_dpi_cor_100dpi.png'], remove_text=True, tol=0 if platform.machine() == 'x86_64' else 0.02, savefig_kwarg=dict(dpi=100))
def test_fancyarrow_dpi_cor_100dpi():
    """
    Check the export of a FancyArrowPatch @ 100 DPI. FancyArrowPatch is
    instantiated through a dedicated function because another similar test
    checks a similar export but with a different DPI value.

    Remark: test only a rasterized format.
    """
    __prepare_fancyarrow_dpi_cor_test()