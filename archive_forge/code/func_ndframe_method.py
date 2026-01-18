import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.fixture(params=_all_methods, ids=lambda x: idfn(x[-1]))
def ndframe_method(request):
    """
    An NDFrame method returning an NDFrame.
    """
    return request.param