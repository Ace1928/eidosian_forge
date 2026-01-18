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
def doc_from_data(document_element_name, data, extended_properties=None):
    doc = AzureXmlSerializer.doc_from_xml(document_element_name)
    AzureXmlSerializer.data_to_xml(data, doc)
    if extended_properties is not None:
        doc.append(AzureXmlSerializer.extended_properties_dict_to_xml_fragment(extended_properties))
    result = ensure_string(ET.tostring(doc, encoding='utf-8'))
    return result