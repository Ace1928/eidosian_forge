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
def extended_properties_dict_to_xml_fragment(extended_properties):
    if extended_properties is not None and len(extended_properties) > 0:
        xml = ET.Element('ExtendedProperties')
        for key, val in extended_properties.items():
            extended_property = ET.Element('ExtendedProperty')
            name = ET.Element('Name')
            name.text = _str(key)
            value = ET.Element('Value')
            value.text = _str(val)
            extended_property.append(name)
            extended_property.append(value)
            xml.append(extended_property)
        return xml