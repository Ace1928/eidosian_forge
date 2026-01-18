from __future__ import absolute_import
import copy
import datetime
import pickle
import hypothesis
import hypothesis.extra.pytz
import hypothesis.strategies
import pytest
from . import iso8601
@hypothesis.given(s=hypothesis.strategies.datetimes(timezones=hypothesis.extra.pytz.timezones()))
def test_hypothesis_valid_datetimes_with_timezone(s: datetime.datetime) -> None:
    as_string = s.isoformat()
    parsed = iso8601.parse_date(as_string)
    print(f'{s!r} {as_string!r} {parsed!r}')
    assert s == parsed