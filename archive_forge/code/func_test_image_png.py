import pytest
from rpy2.robjects.packages import PackageNotInstalledError
from rpy2.robjects.vectors import DataFrame
@pytest.mark.skipif(not has_ggplot2, reason=msg)
def test_image_png():
    dataf = DataFrame({'x': 1, 'Y': 2})
    g = rpy2.robjects.lib.ggplot2.ggplot(dataf)
    img = ggplot.image_png(g)
    assert img