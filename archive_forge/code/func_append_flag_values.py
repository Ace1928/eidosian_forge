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
def append_flag_values(self, flag_values):
    """Appends flags registered in another FlagValues instance.

    Args:
      flag_values: FlagValues, the FlagValues instance from which to copy flags.
    """
    for flag_name, flag in six.iteritems(flag_values._flags()):
        if flag_name == flag.name:
            try:
                self[flag_name] = flag
            except _exceptions.DuplicateFlagError:
                raise _exceptions.DuplicateFlagError.from_flag(flag_name, self, other_flag_values=flag_values)