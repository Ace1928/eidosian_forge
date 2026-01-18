from __future__ import unicode_literals
from past.builtins import basestring
from ._utils import basestring
from .nodes import (
@output_operator()
def overwrite_output(stream):
    """Overwrite output files without asking (ffmpeg ``-y`` option)

    Official documentation: `Main options <https://ffmpeg.org/ffmpeg.html#Main-options>`__
    """
    return GlobalNode(stream, overwrite_output.__name__, ['-y']).stream()