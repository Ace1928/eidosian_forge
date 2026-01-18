import base64
import os
import re
import textwrap
import warnings
from urllib.parse import quote
from xml.etree.ElementTree import Element
import bleach
from defusedxml import ElementTree  # type:ignore[import-untyped]
from nbconvert.preprocessors.sanitize import _get_default_css_sanitizer
def posix_path(path):
    """Turn a path into posix-style path/to/etc

    Mainly for use in latex on Windows,
    where native Windows paths are not allowed.
    """
    if os.path.sep != '/':
        return path.replace(os.path.sep, '/')
    return path