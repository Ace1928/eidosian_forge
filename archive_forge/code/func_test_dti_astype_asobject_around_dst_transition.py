from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_astype_asobject_around_dst_transition(self, tzstr):
    rng = date_range('2/13/2010', '5/6/2010', tz=tzstr)
    objs = rng.astype(object)
    for i, x in enumerate(objs):
        exval = rng[i]
        assert x == exval
        assert x.tzinfo == exval.tzinfo
    objs = rng.astype(object)
    for i, x in enumerate(objs):
        exval = rng[i]
        assert x == exval
        assert x.tzinfo == exval.tzinfo