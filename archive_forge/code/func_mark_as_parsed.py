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
def mark_as_parsed(self):
    """Explicitly marks flags as parsed.

    Use this when the caller knows that this FlagValues has been parsed as if
    a __call__() invocation has happened.  This is only a public method for
    use by things like appcommands which do additional command like parsing.
    """
    self.__dict__['__flags_parsed'] = True