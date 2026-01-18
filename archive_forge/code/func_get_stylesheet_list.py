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
def get_stylesheet_list(settings):
    """
    Retrieve list of stylesheet references from the settings object.
    """
    assert not (settings.stylesheet and settings.stylesheet_path), 'stylesheet and stylesheet_path are mutually exclusive.'
    stylesheets = settings.stylesheet_path or settings.stylesheet or []
    if not isinstance(stylesheets, list):
        stylesheets = [path.strip() for path in stylesheets.split(',')]
    return [find_file_in_dirs(path, settings.stylesheet_dirs) for path in stylesheets]