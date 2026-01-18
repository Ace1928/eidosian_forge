import pytest
from rpy2.robjects import packages
from rpy2.robjects import rl
def test_aes(self):
    gp = ggplot2.ggplot(mtcars)
    gp += ggplot2.aes(x='wt', y='mpg')
    gp += ggplot2.geom_point()
    assert isinstance(gp, ggplot2.GGPlot)
    gp = ggplot2.ggplot(mtcars)
    gp += ggplot2.aes('wt', 'mpg')
    gp += ggplot2.geom_point()
    assert isinstance(gp, ggplot2.GGPlot)