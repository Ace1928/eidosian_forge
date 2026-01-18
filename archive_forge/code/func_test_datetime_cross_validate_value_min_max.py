import pytest
from contextlib import nullcontext
import datetime
import itertools
import pytz
from traitlets import TraitError
from ..widget_datetime import DatetimePicker
@pytest.mark.parametrize('input_value,input_min,input_max,expected', _permuted_dts())
def test_datetime_cross_validate_value_min_max(input_value, input_min, input_max, expected):
    w = DatetimePicker(value=dt_2002, min=dt_2002, max=dt_2002)
    should_raise = expected is TraitError
    with pytest.raises(expected) if should_raise else nullcontext():
        with w.hold_trait_notifications():
            w.value = input_value
            w.min = input_min
            w.max = input_max
    if not should_raise:
        assert w.value is expected