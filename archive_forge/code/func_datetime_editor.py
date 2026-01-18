import datetime
from functools import partial
import logging
def datetime_editor():
    """ Factory function that returns an editor with date & time for
    editing Datetime values.
    """
    from traitsui.api import DatetimeEditor
    return DatetimeEditor()