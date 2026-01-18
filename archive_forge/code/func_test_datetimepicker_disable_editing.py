import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_disable_editing(page):
    datetime_picker_widget = DatetimePicker(disabled=True)
    serve_component(page, datetime_picker_widget)
    expect(page.locator('.flatpickr-input')).to_have_attribute('disabled', 'true')