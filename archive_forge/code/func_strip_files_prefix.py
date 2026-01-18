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
def strip_files_prefix(text):
    """
    Fix all fake URLs that start with ``files/``, stripping out the ``files/`` prefix.
    Applies to both urls (for html) and relative paths (for markdown paths).

    Parameters
    ----------
    text : str
        Text in which to replace 'src="files/real...' with 'src="real...'
    """
    cleaned_text = files_url_pattern.sub('\\1=\\2', text)
    cleaned_text = markdown_url_pattern.sub('\\1[\\2](\\3)', cleaned_text)
    return cleaned_text