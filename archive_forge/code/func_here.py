from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def here(self, keepQuery=False):
    """
        Get the current directory of this L{URLPath}.

        @param keepQuery: Whether to keep the query parameters on the returned
            L{URLPath}.
        @type keepQuery: L{bool}

        @return: a new L{URLPath}
        """
    return self._mod(self._url.click('.'), keepQuery)