import re
import xml.etree.ElementTree as etree
from io import BytesIO
from copy import deepcopy
from time import sleep
from base64 import b64encode
from typing import Dict
from functools import wraps
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
def handle_map(map, name):
    tmp = {}
    types = [type(x) for x in map.values()]
    if XmlListConfig not in types and XmlDictConfig not in types and (dict not in types):
        return map
    elif XmlListConfig in types:
        result = handle_seq(map, name)
        return result
    else:
        for k, v in map.items():
            if isinstance(v, str):
                tmp.update({k: v})
            if isinstance(v, dict):
                cls = build_class(k.capitalize(), v)
                tmp.update({k: cls})
            elif isinstance(v, XmlDictConfig):
                cls = build_class(k.capitalize(), v)
                return (k, cls)
        return tmp