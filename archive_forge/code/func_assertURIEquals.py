from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
def assertURIEquals(self, uri, scheme, netloc, host, port, path, params=b'', query=b'', fragment=b''):
    """
        Assert that all of a L{client.URI}'s components match the expected
        values.

        @param uri: U{client.URI} instance whose attributes will be checked
            for equality.

        @type scheme: L{bytes}
        @param scheme: URI scheme specifier.

        @type netloc: L{bytes}
        @param netloc: Network location component.

        @type host: L{bytes}
        @param host: Host name.

        @type port: L{int}
        @param port: Port number.

        @type path: L{bytes}
        @param path: Hierarchical path.

        @type params: L{bytes}
        @param params: Parameters for last path segment, defaults to C{b''}.

        @type query: L{bytes}
        @param query: Query string, defaults to C{b''}.

        @type fragment: L{bytes}
        @param fragment: Fragment identifier, defaults to C{b''}.
        """
    self.assertEqual((scheme, netloc, host, port, path, params, query, fragment), (uri.scheme, uri.netloc, uri.host, uri.port, uri.path, uri.params, uri.query, uri.fragment))