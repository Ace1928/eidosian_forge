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
def comment_lines(text, prefix='# '):
    """
    Build a Python comment line from input text.

    Parameters
    ----------
    text : str
        Text to comment out.
    prefix : str
        Character to append to the start of each line.
    """
    return prefix + ('\n' + prefix).join(text.split('\n'))