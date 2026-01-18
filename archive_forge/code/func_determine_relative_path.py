import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def determine_relative_path(from_path, to_path):
    """Determine a relative path from from_path to to_path."""
    from_segments = osutils.splitpath(from_path)
    to_segments = osutils.splitpath(to_path)
    count = -1
    for count, (from_element, to_element) in enumerate(zip(from_segments, to_segments)):
        if from_element != to_element:
            break
    else:
        count += 1
    unique_from = from_segments[count:]
    unique_to = to_segments[count:]
    segments = ['..'] * len(unique_from) + unique_to
    if len(segments) == 0:
        return '.'
    return osutils.pathjoin(*segments)