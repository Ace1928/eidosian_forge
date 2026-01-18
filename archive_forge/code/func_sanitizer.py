import datetime
from lxml import etree
from oslo_log import log as logging
from oslo_serialization import jsonutils
def sanitizer(obj):
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return str(obj)