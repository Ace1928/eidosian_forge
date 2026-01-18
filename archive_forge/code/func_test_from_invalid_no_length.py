import subprocess
import pytest
import sys
import os
import textwrap
import rpy2.rinterface as ri
def test_from_invalid_no_length():
    s = (x for x in range(30))
    with pytest.raises(TypeError):
        ri.vector(s, ri.RTYPES.INTSXP)