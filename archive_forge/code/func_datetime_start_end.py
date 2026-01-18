import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
@pytest.fixture
def datetime_start_end():
    start = datetime.datetime(2021, 3, 2)
    end = datetime.datetime(2021, 3, 3)
    selectable_dates = ['March 2, 2021', 'March 3, 2021']
    return (start, end, selectable_dates)