import json
import six
def loadfile(filename, cache=None):
    """Loading of client_secrets JSON file, optionally backed by a cache.

    Typical cache storage would be App Engine memcache service,
    but you can pass in any other cache client that implements
    these methods:

    * ``get(key, namespace=ns)``
    * ``set(key, value, namespace=ns)``

    Usage::

        # without caching
        client_type, client_info = loadfile('secrets.json')
        # using App Engine memcache service
        from google.appengine.api import memcache
        client_type, client_info = loadfile('secrets.json', cache=memcache)

    Args:
        filename: string, Path to a client_secrets.json file on a filesystem.
        cache: An optional cache service client that implements get() and set()
        methods. If not specified, the file is always being loaded from
                 a filesystem.

    Raises:
        InvalidClientSecretsError: In case of a validation error or some
                                   I/O failure. Can happen only on cache miss.

    Returns:
        (client_type, client_info) tuple, as _loadfile() normally would.
        JSON contents is validated only during first load. Cache hits are not
        validated.
    """
    _SECRET_NAMESPACE = 'oauth2client:secrets#ns'
    if not cache:
        return _loadfile(filename)
    obj = cache.get(filename, namespace=_SECRET_NAMESPACE)
    if obj is None:
        client_type, client_info = _loadfile(filename)
        obj = {client_type: client_info}
        cache.set(filename, obj, namespace=_SECRET_NAMESPACE)
    return next(six.iteritems(obj))