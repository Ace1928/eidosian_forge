import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetime_picker_start_end_datetime64(page):
    datetime_picker_widget = DatetimePicker(value=datetime.datetime(2021, 3, 2), start=np.datetime64('2021-03-02'), end=np.datetime64('2021-03-03'))
    serve_component(page, datetime_picker_widget)
    datetime_picker = page.locator('.flatpickr-input')
    datetime_picker.dblclick()
    prev_month_day = page.locator('[aria-label="March 1, 2021"]')
    assert 'flatpickr-disabled' in prev_month_day.get_attribute('class'), 'The date should be disabled'
    next_month_day = page.locator('[aria-label="March 3, 2021"]')
    assert 'flatpickr-disabled' not in next_month_day.get_attribute('class'), 'The date should be enabled'
    next_next_month_day = page.locator('[aria-label="March 4, 2021"]')
    assert 'flatpickr-disabled' in next_next_month_day.get_attribute('class'), 'The date should be disabled'