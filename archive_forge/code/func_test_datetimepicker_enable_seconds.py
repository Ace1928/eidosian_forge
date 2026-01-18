import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_enable_seconds(page):
    datetime_picker_widget = DatetimePicker(enable_seconds=False)
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    datetime_value.dblclick()
    time_editor_with_sec = page.locator('.flatpickr-calendar .flatpickr-time.time24hr.hasSeconds')
    expect(time_editor_with_sec).to_have_count(0)
    time_editor_with_sec = page.locator('.flatpickr-calendar .flatpickr-time.time24hr')
    expect(time_editor_with_sec).to_have_count(1)
    time_inputs = page.locator('.flatpickr-calendar .flatpickr-time .numInputWrapper')
    time_up_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowUp')
    time_down_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowDown')
    expect(time_inputs).to_have_count(2)
    expect(time_up_buttons).to_have_count(2)
    expect(time_down_buttons).to_have_count(2)
    time_separators = page.locator('.flatpickr-calendar .flatpickr-time .flatpickr-time-separator')
    expect(time_separators).to_have_count(1)