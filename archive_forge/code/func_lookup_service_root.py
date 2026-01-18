from urllib.parse import urlparse, urlunparse
def lookup_service_root(root):
    """Dereference an alias to a service root.

    A recognized server alias such as "staging" gets turned into the
    appropriate URI. A URI gets returned as is. Any other string raises a
    ValueError.
    """
    if root in service_roots:
        return service_roots[root]
    scheme, netloc, path, parameters, query, fragment = urlparse(root)
    if scheme != '' and netloc != '':
        return root
    raise ValueError('%s is not a valid URL or an alias for any Launchpad server' % root)