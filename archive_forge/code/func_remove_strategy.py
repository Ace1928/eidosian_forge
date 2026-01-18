from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
def remove_strategy(self, domain):
    """Remove the domain and strategy from the collection of strategies.

        :param str domain: The domain you wish remove. For example,
            ``'https://api.github.com'``.

        .. code-block:: python

            a = AuthHandler({'example.com', ('foo', 'bar')})
            a.remove_strategy('example.com')
            assert a.strategies == {}

        """
    key = self._key_from_url(domain)
    if key in self.strategies:
        del self.strategies[key]