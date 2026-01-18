import sys
import os
import os.path
import re
import itertools
import warnings
import unicodedata
from docutils import ApplicationError, DataError, __version_info__
from docutils import nodes
from docutils.nodes import unescape
import docutils.io
from docutils.utils.error_reporting import ErrorOutput, SafeString
def get_trim_footnote_ref_space(settings):
    """
    Return whether or not to trim footnote space.

    If trim_footnote_reference_space is not None, return it.

    If trim_footnote_reference_space is None, return False unless the
    footnote reference style is 'superscript'.
    """
    if settings.trim_footnote_reference_space is None:
        return hasattr(settings, 'footnote_references') and settings.footnote_references == 'superscript'
    else:
        return settings.trim_footnote_reference_space