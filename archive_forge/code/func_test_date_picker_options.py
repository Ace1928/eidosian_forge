from datetime import date, datetime
from pathlib import Path
import numpy as np
import pytest
from bokeh.models.widgets import FileInput as BkFileInput
from panel import config
from panel.widgets import (
def test_date_picker_options(document, comm):
    options = [date(2018, 9, 1), date(2018, 9, 2), date(2018, 9, 3)]
    datetime_picker = DatePicker(name='DatetimePicker', value=date(2018, 9, 2), options=options)
    assert datetime_picker.value == date(2018, 9, 2)
    assert datetime_picker.enabled_dates == options