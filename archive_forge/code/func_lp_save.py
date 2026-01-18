from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
def lp_save(self):
    """Save changes to the entry."""
    representation = self._transform_resources_to_links(self._dirty_attributes)
    headers = {}
    etag = getattr(self, 'http_etag', None)
    if etag is not None:
        headers['If-Match'] = etag
    response, content = self._root._browser.patch(URI(self.self_link), representation, headers)
    if response.status == 301:
        self.lp_refresh(response['location'])
    self._dirty_attributes.clear()
    content_type = response['content-type']
    if response.status == 209 and content_type == self.JSON_MEDIA_TYPE:
        if isinstance(content, binary_type):
            content = content.decode('utf-8')
        new_representation = loads(content)
        self._wadl_resource.representation = new_representation
        self._wadl_resource.media_type = content_type