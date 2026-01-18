from collections import namedtuple
from . import compat
from . import exceptions
from . import misc
from . import normalizers
from . import uri
@classmethod
def from_parts(cls, scheme=None, userinfo=None, host=None, port=None, path=None, query=None, fragment=None, encoding='utf-8', lazy_normalize=True):
    """Create a ParseResult instance from its parts."""
    authority = ''
    if userinfo is not None:
        authority += userinfo + '@'
    if host is not None:
        authority += host
    if port is not None:
        authority += ':{0}'.format(int(port))
    uri_ref = uri.URIReference(scheme=scheme, authority=authority, path=path, query=query, fragment=fragment, encoding=encoding)
    if not lazy_normalize:
        uri_ref = uri_ref.normalize()
    to_bytes = compat.to_bytes
    userinfo, host, port = authority_from(uri_ref, strict=True)
    return cls(scheme=to_bytes(scheme, encoding), userinfo=to_bytes(userinfo, encoding), host=to_bytes(host, encoding), port=port, path=to_bytes(path, encoding), query=to_bytes(query, encoding), fragment=to_bytes(fragment, encoding), uri_ref=uri_ref, encoding=encoding, lazy_normalize=lazy_normalize)