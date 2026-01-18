import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
@max_concurrent_streams.setter
def max_concurrent_streams(self, value):
    self[SettingCodes.MAX_CONCURRENT_STREAMS] = value