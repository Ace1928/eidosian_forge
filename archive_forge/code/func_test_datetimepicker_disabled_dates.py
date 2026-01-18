import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_disabled_dates(page, disabled_dates):
    active_date, disabled_list, disabled_str_list = disabled_dates
    datetime_picker_widget = DatetimePicker(disabled_dates=disabled_list, value=active_date)
    serve_component(page, datetime_picker_widget)
    datetime_value = page.locator('.flatpickr-input')
    datetime_value.dblclick()
    disabled_days = page.locator('.flatpickr-calendar .flatpickr-day.flatpickr-disabled')
    expect(disabled_days).to_have_count(len(disabled_list))
    for i in range(len(disabled_list)):
        assert disabled_days.nth(i).get_attribute('aria-label') == disabled_str_list[i]