import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def datetime_value_data():
    year, month, day, hour, min, sec = (2021, 3, 2, 23, 55, 0)
    month_str = 'March'
    date_str = 'March 2, 2021'
    datetime_str = '2021-03-02 23:55:00'
    return (year, month, day, hour, min, sec, month_str, date_str, datetime_str)