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
def flags_into_string(self):
    """Returns a string with the flags assignments from this FlagValues object.

    This function ignores flags whose value is None.  Each flag
    assignment is separated by a newline.

    NOTE: MUST mirror the behavior of the C++ CommandlineFlagsIntoString
    from https://github.com/gflags/gflags.

    Returns:
      str, the string with the flags assignments from this FlagValues object.
      The flags are ordered by (module_name, flag_name).
    """
    module_flags = sorted(self.flags_by_module_dict().items())
    s = ''
    for unused_module_name, flags in module_flags:
        flags = sorted(flags, key=lambda f: f.name)
        for flag in flags:
            if flag.value is not None:
                s += flag.serialize() + '\n'
    return s