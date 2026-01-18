import re
import math
import textwrap
import six
from wcwidth import wcwidth
from blessed._capabilities import CAPABILITIES_CAUSE_MOVEMENT
@property
def named_pattern(self):
    """Regular expression pattern for capability with named group."""
    return '(?P<{self.name}>{self.pattern})'.format(self=self)