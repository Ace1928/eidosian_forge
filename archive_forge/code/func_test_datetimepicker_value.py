import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_value(page, march_2021, datetime_value_data):
    year, month, day, hour, min, sec, month_str, date_str, datetime_str = datetime_value_data
    march_2021_str, num_days, num_prev_month_days, num_next_month_days = march_2021
    datetime_picker_widget = DatetimePicker(value=datetime.datetime(year, month, day, hour, min, sec))
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    assert datetime_value.input_value() == datetime_str
    datetime_value.dblclick()
    year_input = page.locator('.flatpickr-current-month .numInput.cur-year')
    assert int(year_input.input_value()) == year
    days_container = page.locator('.flatpickr-calendar .flatpickr-days .dayContainer')
    expect(days_container).to_have_text(march_2021_str, use_inner_text=True)
    prev_month_days = page.locator('.flatpickr-calendar .flatpickr-day.prevMonthDay')
    expect(prev_month_days).to_have_count(num_prev_month_days)
    next_month_days = page.locator('.flatpickr-calendar .flatpickr-day.nextMonthDay')
    expect(next_month_days).to_have_count(num_next_month_days)
    all_days = page.locator('.flatpickr-calendar .flatpickr-day')
    expect(all_days).to_have_count(num_days)
    selected_day = page.locator('.flatpickr-calendar .flatpickr-day.selected')
    assert selected_day.get_attribute('aria-label') == date_str
    hour_input = page.locator('.flatpickr-calendar .flatpickr-time .numInput.flatpickr-hour')
    assert int(hour_input.input_value()) == hour
    min_input = page.locator('.flatpickr-calendar .flatpickr-time .numInput.flatpickr-minute')
    assert int(min_input.input_value()) == min
    sec_input = page.locator('.flatpickr-calendar .flatpickr-time .numInput.flatpickr-second')
    assert int(sec_input.input_value()) == sec
    time_up_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowUp')
    time_down_buttons = page.locator('.flatpickr-calendar .flatpickr-time .arrowDown')
    hour_up = time_up_buttons.nth(0)
    min_up = time_up_buttons.nth(1)
    sec_down = time_down_buttons.nth(2)
    hour_before_click = hour_input.input_value()
    hour_up.click()
    hour_after_click = hour_input.input_value()
    assert valid_next_time(hour_before_click, hour_after_click, max_value=24, amount=1)
    min_before_click = min_input.input_value()
    min_up.click()
    min_after_click = min_input.input_value()
    assert valid_next_time(min_before_click, min_after_click, max_value=60, amount=5)
    sec_before_click = sec_input.input_value()
    sec_down.click()
    sec_after_click = sec_input.input_value()
    assert valid_prev_time(sec_before_click, sec_after_click, max_value=60, amount=5)