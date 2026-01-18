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
def new_reporter(source_path, settings):
    """
    Return a new Reporter object.

    :Parameters:
        `source` : string
            The path to or description of the source text of the document.
        `settings` : optparse.Values object
            Runtime settings.
    """
    reporter = Reporter(source_path, settings.report_level, settings.halt_level, stream=settings.warning_stream, debug=settings.debug, encoding=settings.error_encoding, error_handler=settings.error_encoding_error_handler)
    return reporter