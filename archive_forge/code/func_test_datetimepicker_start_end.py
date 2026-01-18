import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_start_end(page, march_2021, datetime_start_end):
    start, end, selectable_dates = datetime_start_end
    march_2021_str, num_days, _, _ = march_2021
    datetime_picker_widget = DatetimePicker(start=start, end=end)
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    datetime_value.dblclick()
    days_container = page.locator('.flatpickr-calendar .flatpickr-days .dayContainer')
    expect(days_container).to_have_text(march_2021_str, use_inner_text=True)
    disabled_days = page.locator('.flatpickr-calendar .flatpickr-day.flatpickr-disabled')
    expect(disabled_days).to_have_count(num_days - len(selectable_dates))