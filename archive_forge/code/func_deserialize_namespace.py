import collections
from ._caveat import error_caveat
from ._utils import condition_with_prefix
def deserialize_namespace(data):
    """ Deserialize a Namespace object.

    :param data: bytes or str
    :return: namespace
    """
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    kvs = data.split()
    uri_to_prefix = {}
    for kv in kvs:
        i = kv.rfind(':')
        if i == -1:
            raise ValueError('no colon in namespace field {}'.format(repr(kv)))
        uri, prefix = (kv[0:i], kv[i + 1:])
        if not is_valid_schema_uri(uri):
            raise ValueError('invalid URI {} in namespace field {}'.format(repr(uri), repr(kv)))
        if not is_valid_prefix(prefix):
            raise ValueError('invalid prefix {} in namespace field {}'.format(repr(prefix), repr(kv)))
        if uri in uri_to_prefix:
            raise ValueError('duplicate URI {} in namespace {}'.format(repr(uri), repr(data)))
        uri_to_prefix[uri] = prefix
    return Namespace(uri_to_prefix)