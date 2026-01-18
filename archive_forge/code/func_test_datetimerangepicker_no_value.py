import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimerangepicker_no_value(page, datetime_start_end):
    datetime_picker_widget = DatetimeRangePicker()
    serve_component(page, datetime_picker_widget)
    datetime_picker = page.locator('.flatpickr-input')
    assert datetime_picker.input_value() == ''
    datetime_picker_widget.value = datetime_start_end[:2]
    expected = '2021-03-02 00:00:00 to 2021-03-03 00:00:00'
    wait_until(lambda: datetime_picker.input_value() == expected, page)
    datetime_picker_widget.value = None
    wait_until(lambda: datetime_picker.input_value() == '', page)