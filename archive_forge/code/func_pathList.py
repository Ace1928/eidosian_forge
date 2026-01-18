from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def pathList(self, unquote=False, copy=True):
    """
        Split this URL's path into its components.

        @param unquote: whether to remove %-encoding from the returned strings.

        @param copy: (ignored, do not use)

        @return: The components of C{self.path}
        @rtype: L{list} of L{bytes}
        """
    segments = self._url.path
    mapper = lambda x: x.encode('ascii')
    if unquote:
        mapper = lambda x, m=mapper: m(urlunquote(x))
    return [b''] + [mapper(segment) for segment in segments]