from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
@property
def member_names(self):
    """The accepted enum names, in lowercase if not case sensitive."""
    return self._member_names