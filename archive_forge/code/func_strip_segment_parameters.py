import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def strip_segment_parameters(url):
    """Strip the segment parameters from a URL.

    Args:
      url: A relative or absolute URL
    Returns: url
    """
    base_url, subsegments = split_segment_parameters_raw(url)
    return base_url