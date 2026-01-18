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
def prevent_list_blocks(s):
    """
    Prevent presence of enumerate or itemize blocks in latex headings cells
    """
    out = re.sub('(^\\s*\\d*)\\.', '\\1\\.', s)
    out = re.sub('(^\\s*)\\-', '\\1\\-', out)
    out = re.sub('(^\\s*)\\+', '\\1\\+', out)
    out = re.sub('(^\\s*)\\*', '\\1\\*', out)
    return out