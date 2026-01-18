import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
@initial_window_size.setter
def initial_window_size(self, value):
    self[SettingCodes.INITIAL_WINDOW_SIZE] = value