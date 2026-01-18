import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def decode_map_element(self, item_type, value):
    """Decode a single element for a map"""
    import urllib
    key = value
    if ':' in value:
        key, value = value.split(':', 1)
        key = urllib.unquote(key)
    if self.model_class in item_type.mro():
        value = item_type(id=value)
    else:
        value = self.decode(item_type, value)
    return (key, value)