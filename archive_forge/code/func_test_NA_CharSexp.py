import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
def test_NA_CharSexp():
    with (robjects.default_converter + rpyn.converter).context():
        values = robjects.r('c(NA_character_)')
    assert values[0] is None