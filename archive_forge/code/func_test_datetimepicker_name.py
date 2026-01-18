import datetime
import numpy as np
import pytest
from playwright.sync_api import expect
from panel.tests.util import serve_component, wait_until
from panel.widgets import DatetimePicker, DatetimeRangePicker, TextAreaInput
def test_datetimepicker_name(page):
    name = 'Datetime Picker'
    datetime_picker_widget = DatetimePicker(name=name, css_classes=['datetimepicker-with-name'])
    serve_component(page, datetime_picker_widget)
    expect(page.locator('.datetimepicker-with-name > .bk-input-group > label')).to_have_text(name)