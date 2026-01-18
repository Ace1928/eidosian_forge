import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_float_nan():
    with pytest.raises(ValueError):
        ri.vector(['a'], ri.RTYPES.REALSXP)