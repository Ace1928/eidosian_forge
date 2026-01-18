import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
def horizontal_distance(self, text):
    """
        Horizontal carriage adjusted by capability, may be negative.

        :rtype: int
        :arg str text: for capabilities *parm_left_cursor*,
            *parm_right_cursor*, provide the matching sequence
            text, its interpreted distance is returned.

        :returns: 0 except for matching '
        """
    value = {'cursor_left': -1, 'backspace': -1, 'cursor_right': 1, 'tab': 8, 'ascii_tab': 8}.get(self.name)
    if value is not None:
        return value
    unit = {'parm_left_cursor': -1, 'parm_right_cursor': 1}.get(self.name)
    if unit is not None:
        value = int(self.re_compiled.match(text).group(1))
        return unit * value
    return 0