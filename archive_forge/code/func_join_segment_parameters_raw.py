import os
import posixpath
import re
import sys
from typing import Tuple, Union
from urllib import parse as urlparse
from . import errors, osutils
def join_segment_parameters_raw(base, *subsegments):
    """Create a new URL by adding subsegments to an existing one.

    This adds the specified subsegments to the last path in the specified
    base URL. The subsegments should be bytestrings.

    :note: You probably want to use join_segment_parameters instead.
    """
    if not subsegments:
        return base
    for subsegment in subsegments:
        if not isinstance(subsegment, str):
            raise TypeError('Subsegment %r is not a bytestring' % subsegment)
        if ',' in subsegment:
            raise InvalidURLJoin(', exists in subsegments', base, subsegments)
    return ','.join((base,) + subsegments)