from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
def get_str_bbox(self, s):
    """Return the string bounding box."""
    return self.get_str_bbox_and_descent(s)[:4]