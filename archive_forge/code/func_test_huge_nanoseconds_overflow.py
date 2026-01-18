import re
import numpy as np
import pytest
from pandas._libs.tslibs.timedeltas import (
from pandas import (
import pandas._testing as tm
def test_huge_nanoseconds_overflow():
    assert delta_to_nanoseconds(Timedelta(10000000000.0)) == 10000000000.0
    assert delta_to_nanoseconds(Timedelta(nanoseconds=10000000000.0)) == 10000000000.0