import numpy as np
import param
import pytest
from packaging.version import Version
from holoviews import Annotation, Arrow, HLine, Spline, Text, VLine
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
def test_hline_invalid_constructor(self):
    if Version(param.__version__) > Version('2.0.0a2'):
        err = "ClassSelector parameter 'HLine.y' value must be an instance of"
    else:
        err = "ClassSelector parameter 'y' value must be an instance of"
    with pytest.raises(ValueError) as excinfo:
        HLine(None)
    assert err in str(excinfo.value)