import re
from typing import Dict, Union
from xml.etree.ElementTree import (Element, ElementTree, ParseError,
from .. import errors, lazy_regex
from . import inventory, serializer
def get_utf8_or_ascii(a_str):
    """Return a cached version of the string.

    cElementTree will return a plain string if the XML is plain ascii. It only
    returns Unicode when it needs to. We want to work in utf-8 strings. So if
    cElementTree returns a plain string, we can just return the cached version.
    If it is Unicode, then we need to encode it.

    :param a_str: An 8-bit string or Unicode as returned by
                  cElementTree.Element.get()
    :return: A utf-8 encoded 8-bit string.
    """
    if a_str.__class__ is str:
        return a_str.encode('utf-8')
    else:
        return a_str