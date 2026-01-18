from collections import abc
import xml.sax
import hashlib
import string
from boto.connection import AWSQueryConnection
from boto.exception import BotoServerError
import boto.mws.exception
import boto.mws.response
from boto.handler import XmlHandler
from boto.compat import filter, map, six, encodebytes
def method_for(self, name):
    """Return the MWS API method referred to in the argument.
           The named method can be in CamelCase or underlined_lower_case.
           This is the complement to MWSConnection.any_call.action
        """
    action = '_' in name and string.capwords(name, '_') or name
    if action in api_call_map:
        return getattr(self, api_call_map[action])
    return None