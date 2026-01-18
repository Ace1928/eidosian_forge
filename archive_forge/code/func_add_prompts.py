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
def add_prompts(code, first='>>> ', cont='... '):
    """Add prompts to code snippets"""
    new_code = []
    code_list = code.split('\n')
    new_code.append(first + code_list[0])
    for line in code_list[1:]:
        new_code.append(cont + line)
    return '\n'.join(new_code)