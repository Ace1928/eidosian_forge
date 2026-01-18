import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_military_time(page):
    datetime_picker_widget = DatetimePicker(military_time=False)
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    datetime_value.dblclick()
    time_editor_with_sec = page.locator('.flatpickr-calendar .flatpickr-time.time24hr.hasSeconds')
    expect(time_editor_with_sec).to_have_count(0)
    time_editor_with_sec = page.locator('.flatpickr-calendar .flatpickr-time.hasSeconds')
    expect(time_editor_with_sec).to_have_count(1)
    time_inputs = page.locator('.flatpickr-calendar .flatpickr-time .numInputWrapper')
    time_am_pm = page.locator('.flatpickr-calendar .flatpickr-time .flatpickr-am-pm')
    time_up_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowUp')
    time_down_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowDown')
    expect(time_inputs).to_have_count(3)
    expect(time_up_buttons).to_have_count(3)
    expect(time_down_buttons).to_have_count(3)
    expect(time_am_pm).to_have_count(1)
    time_separators = page.locator('.flatpickr-calendar .flatpickr-time .flatpickr-time-separator')
    expect(time_separators).to_have_count(2)
    hour_input = page.locator('.flatpickr-calendar .flatpickr-time .numInput.flatpickr-hour')
    hour_up = time_up_buttons.nth(0)
    hour_down = time_down_buttons.nth(0)
    hour_before_click = hour_input.input_value()
    hour_down.click()
    hour_after_click = hour_input.input_value()
    assert valid_prev_time(hour_before_click, hour_after_click, max_value=13, amount=1)
    hour_before_click = hour_input.input_value()
    hour_up.click()
    hour_after_click = hour_input.input_value()
    assert valid_next_time(hour_before_click, hour_after_click, max_value=13, amount=1)