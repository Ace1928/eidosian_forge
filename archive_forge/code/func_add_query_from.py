from . import compat
from . import normalizers
from . import uri
from . import uri_reference
def add_query_from(self, query_items):
    """Generate and add a query a dictionary or list of tuples.

        .. code-block:: python

            >>> URIBuilder().add_query_from({'a': 'b c'})
            URIBuilder(scheme=None, userinfo=None, host=None, port=None,
                    path=None, query='a=b+c', fragment=None)

            >>> URIBuilder().add_query_from([('a', 'b c')])
            URIBuilder(scheme=None, userinfo=None, host=None, port=None,
                    path=None, query='a=b+c', fragment=None)

        """
    query = normalizers.normalize_query(compat.urlencode(query_items))
    return URIBuilder(scheme=self.scheme, userinfo=self.userinfo, host=self.host, port=self.port, path=self.path, query=query, fragment=self.fragment)