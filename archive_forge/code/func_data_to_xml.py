import re
import copy
import time
import base64
import random
import collections
from xml.dom import minidom
from datetime import datetime
from xml.sax.saxutils import escape as xml_escape
from libcloud.utils.py3 import ET, httplib, urlparse
from libcloud.utils.py3 import urlquote as url_quote
from libcloud.utils.py3 import _real_unicode, ensure_string
from libcloud.utils.misc import ReprMixin
from libcloud.common.azure import AzureRedirectException, AzureServiceManagementConnection
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import NodeState
from libcloud.compute.providers import Provider
@staticmethod
def data_to_xml(data, xml=None):
    """
        Creates an xml fragment from the specified data.
           data: Array of tuples, where first: xml element name
                                        second: xml element text
                                        third: conversion function
        """
    for element in data:
        name = element[0]
        val = element[1]
        if len(element) > 2:
            converter = element[2]
        else:
            converter = None
        if val is not None:
            if converter is not None:
                text = _str(converter(_str(val)))
            else:
                text = _str(val)
            entry = ET.Element(name)
            entry.text = text
            if xml is not None:
                xml.append(entry)
            else:
                return entry
    return xml