import copy
import xml.dom
from xml.dom.NodeFilter import NodeFilter
def resolveEntity(self, publicId, systemId):
    assert systemId is not None
    source = DOMInputSource()
    source.publicId = publicId
    source.systemId = systemId
    source.byteStream = self._get_opener().open(systemId)
    source.encoding = self._guess_media_encoding(source)
    import posixpath, urllib.parse
    parts = urllib.parse.urlparse(systemId)
    scheme, netloc, path, params, query, fragment = parts
    if path and (not path.endswith('/')):
        path = posixpath.dirname(path) + '/'
        parts = (scheme, netloc, path, params, query, fragment)
        source.baseURI = urllib.parse.urlunparse(parts)
    return source