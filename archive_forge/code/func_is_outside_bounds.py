from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
def is_outside_bounds(self, val):
    """Returns whether the value is outside the bounds or not."""
    return self.lower_bound is not None and val < self.lower_bound or (self.upper_bound is not None and val > self.upper_bound)