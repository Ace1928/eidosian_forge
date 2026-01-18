import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
def padd(self, strip=False):
    """
        Return non-destructive horizontal movement as destructive spacing.

        :arg bool strip: Strip terminal sequences
        :rtype: str
        :returns: Text adjusted for horizontal movement
        """
    outp = ''
    for text, cap in iter_parse(self._term, self):
        if not cap:
            outp += text
            continue
        value = cap.horizontal_distance(text)
        if value > 0:
            outp += ' ' * value
        elif value < 0:
            outp = outp[:value]
        elif not strip:
            outp += text
    return outp