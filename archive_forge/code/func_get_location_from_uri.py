import logging
import urllib.parse
from oslo_config import cfg
from glance_store import exceptions
from glance_store.i18n import _
def get_location_from_uri(uri, conf=CONF):
    """
    Given a URI, return a Location object that has had an appropriate
    store parse the URI.

    :param uri: A URI that could come from the end-user in the Location
                attribute/header.
    :param conf: The global configuration.

    Example URIs:
        https://user:pass@example.com:80/images/some-id
        http://example.com/123456
        swift://example.com/container/obj-id
        swift://user:account:pass@authurl.com/container/obj-id
        swift+http://user:account:pass@authurl.com/container/obj-id
        file:///var/lib/glance/images/1
        cinder://volume-id
        s3://accesskey:secretkey@s3.amazonaws.com/bucket/key-id
        s3+https://accesskey:secretkey@s3.amazonaws.com/bucket/key-id
    """
    pieces = urllib.parse.urlparse(uri)
    if pieces.scheme not in SCHEME_TO_CLS_MAP.keys():
        raise exceptions.UnknownScheme(scheme=pieces.scheme)
    scheme_info = SCHEME_TO_CLS_MAP[pieces.scheme]
    return Location(pieces.scheme, scheme_info['location_class'], conf, uri=uri)