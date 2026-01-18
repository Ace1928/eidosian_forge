import mimetypes
from time import time as now
from email.utils import formatdate, parsedate_tz, mktime_tz
from urllib.request import AbstractDigestAuthHandler, parse_keqv_list, parse_http_list
from .httpexceptions import HTTPBadRequest
def list_headers(general=None, request=None, response=None, entity=None):
    """ list all headers for a given category """
    if not (general or request or response or entity):
        general = request = response = entity = True
    search = []
    for bool, strval in ((general, 'general'), (request, 'request'), (response, 'response'), (entity, 'entity')):
        if bool:
            search.append(strval)
    return [head for head in _headers.values() if head.category in search]