import os
import six
from genshi.template.base import TemplateError
from genshi.util import LRUCache
@staticmethod
def prefixed(**delegates):
    """Factory for a load function that delegates to other loaders
        depending on the prefix of the requested template path.
        
        The prefix is stripped from the filename when passing on the load
        request to the delegate.
        
        >>> load = prefixed(
        ...     app1 = lambda filename: ('app1', filename, None, None),
        ...     app2 = lambda filename: ('app2', filename, None, None)
        ... )
        >>> print(load('app1/foo.html'))
        ('app1', 'app1/foo.html', None, None)
        >>> print(load('app2/bar.html'))
        ('app2', 'app2/bar.html', None, None)
        
        :param delegates: mapping of path prefixes to loader functions
        :return: the loader function
        :rtype: ``function``
        """

    def _dispatch_by_prefix(filename):
        for prefix, delegate in delegates.items():
            if filename.startswith(prefix):
                if isinstance(delegate, six.string_types):
                    delegate = directory(delegate)
                filepath, _, fileobj, uptodate = delegate(filename[len(prefix):].lstrip('/\\'))
                return (filepath, filename, fileobj, uptodate)
        raise TemplateNotFound(filename, list(delegates.keys()))
    return _dispatch_by_prefix