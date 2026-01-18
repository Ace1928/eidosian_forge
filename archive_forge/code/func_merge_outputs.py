from __future__ import unicode_literals
from past.builtins import basestring
from ._utils import basestring
from .nodes import (
@output_operator()
def merge_outputs(*streams):
    """Include all given outputs in one ffmpeg command line
    """
    return MergeOutputsNode(streams, merge_outputs.__name__).stream()