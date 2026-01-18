import base64
import os
from urllib import error
from urllib import parse
from urllib import request
from openstack import exceptions
def read_url_content(url):
    try:
        content = request.urlopen(url).read()
    except error.URLError:
        raise exceptions.SDKException('Could not fetch contents for %s' % url)
    if content:
        try:
            content = content.decode('utf-8')
        except ValueError:
            content = base64.encodebytes(content)
    return content