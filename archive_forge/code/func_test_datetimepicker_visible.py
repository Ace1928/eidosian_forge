import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_visible(page):
    datetime_picker_widget = DatetimePicker(visible=False, css_classes=['invisible-datetimepicker'])
    serve_component(page, datetime_picker_widget)
    expect(page.locator('.invisible-datetimepicker')).to_have_css('display', 'none')