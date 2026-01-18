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
def dd_object_to_id(obj, obj_type, id_value='id'):
    """
    Takes in a DD object or string and prints out it's id
    This is a helper method, as many of our functions can take either an object
    or a string, and we need an easy way of converting them

    :param obj: The object to get the id for
    :type  obj: ``object``

    :param  func: The function to call, e.g. ex_get_vlan. Note: This
                  function needs to return an object which has ``status``
                  attribute.
    :type   func: ``function``

    :rtype: ``str``
    """
    if isinstance(obj, obj_type):
        return getattr(obj, id_value)
    elif isinstance(obj, basestring):
        return obj
    else:
        raise TypeError('Invalid type {} looking for basestring or {}'.format(type(obj).__name__, obj_type.__name__))