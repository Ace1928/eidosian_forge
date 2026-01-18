import sys
from math import trunc
from typing import (
def meridian(self, hour: int, token: Any) -> Optional[str]:
    """Returns the meridian indicator for a specified hour and format token.

        :param hour: the ``int`` hour of the day.
        :param token: the format token.
        """
    if token == 'a':
        return self.meridians['am'] if hour < 12 else self.meridians['pm']
    if token == 'A':
        return self.meridians['AM'] if hour < 12 else self.meridians['PM']
    return None