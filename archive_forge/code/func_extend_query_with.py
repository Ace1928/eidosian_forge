from . import compat
from . import normalizers
from . import uri
from . import uri_reference
def extend_query_with(self, query_items):
    """Extend the existing query string with the new query items.

        .. versionadded:: 1.5.0

        .. code-block:: python

            >>> URIBuilder(query='a=b+c').extend_query_with({'a': 'b c'})
            URIBuilder(scheme=None, userinfo=None, host=None, port=None,
                    path=None, query='a=b+c&a=b+c', fragment=None)

            >>> URIBuilder(query='a=b+c').extend_query_with([('a', 'b c')])
            URIBuilder(scheme=None, userinfo=None, host=None, port=None,
                    path=None, query='a=b+c&a=b+c', fragment=None)
        """
    original_query_items = compat.parse_qsl(self.query or '')
    if not isinstance(query_items, list):
        query_items = list(query_items.items())
    return self.add_query_from(original_query_items + query_items)