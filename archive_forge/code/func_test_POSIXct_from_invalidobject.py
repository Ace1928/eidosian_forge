import datetime
import pytest
import time
from rpy2 import robjects
import rpy2.robjects.vectors
def test_POSIXct_from_invalidobject():
    x = ['abc', 3]
    with pytest.raises(TypeError):
        robjects.POSIXct(x)