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
def html2text(element):
    """extract inner text from html

    Analog of jQuery's $(element).text()
    """
    if isinstance(element, (str,)):
        try:
            element = ElementTree.fromstring(element)
        except Exception:
            return element
    text = element.text or ''
    for child in element:
        text += html2text(child)
    text += element.tail or ''
    return text