import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_add(self):
    gp = ggplot2.ggplot(mtcars)
    gp += ggplot2.aes_string(x='wt', y='mpg')
    gp += ggplot2.geom_point()
    assert isinstance(gp, ggplot2.GGPlot)