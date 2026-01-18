import sys
from sympy.external.importtools import version_tuple
import pytest
from sympy.core.cache import clear_cache, USE_CACHE
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from sympy.utilities.misc import ARCH
import re
@pytest.fixture(autouse=True, scope='module')
def check_disabled(request):
    if getattr(request.module, 'disabled', False):
        pytest.skip('test requirements not met.')
    elif getattr(request.module, 'ipython', False):
        if version_tuple(pytest.__version__) < version_tuple('2.6.3') and pytest.config.getvalue('-s') != 'no':
            pytest.skip('run py.test with -s or upgrade to newer version.')