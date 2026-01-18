import io
import re
from contextlib import redirect_stdout
import pytest
from numpy.distutils import log
def teardown_module():
    log.set_verbosity(0, force=True)