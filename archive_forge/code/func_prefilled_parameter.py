from collections import defaultdict
import pytest
from modin.config import Parameter
@pytest.fixture
def prefilled_parameter():
    return make_prefilled(str, 'init')