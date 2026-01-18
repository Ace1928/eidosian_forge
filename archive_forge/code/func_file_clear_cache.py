import sys
from sympy.external.importtools import version_tuple
import pytest
from sympy.core.cache import clear_cache, USE_CACHE
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from sympy.utilities.misc import ARCH
import re
@pytest.fixture(autouse=True, scope='module')
def file_clear_cache():
    clear_cache()