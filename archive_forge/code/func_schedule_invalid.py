import numpy
import pytest
from thinc.api import Optimizer, registry
@pytest.fixture(params=[lambda: 'hello', lambda: _test_schedule_invalid(), lambda: (_ for _ in []), lambda: []], scope='function')
def schedule_invalid(request):
    r_func = request.param
    return r_func()