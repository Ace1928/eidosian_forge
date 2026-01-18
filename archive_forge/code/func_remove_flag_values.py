from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
def remove_flag_values(self, flag_values):
    """Remove flags that were previously appended from another FlagValues.

    Args:
      flag_values: FlagValues, the FlagValues instance containing flags to
        remove.
    """
    for flag_name in flag_values:
        self.__delattr__(flag_name)