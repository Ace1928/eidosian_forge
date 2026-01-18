import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_remove_value(page, datetime_start_end):
    datetime_picker_widget = DatetimePicker(value=datetime_start_end[0])
    serve_component(page, datetime_picker_widget)
    datetime_picker = page.locator('.flatpickr-input')
    assert datetime_picker.input_value() == '2021-03-02 00:00:00'
    datetime_picker.dblclick()
    datetime_picker.press('Backspace')
    datetime_picker.press('Escape')
    wait_until(lambda: datetime_picker_widget.value is None, page)