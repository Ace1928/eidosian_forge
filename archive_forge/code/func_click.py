from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
def click(self, st):
    """
        Return a path which is the URL where a browser would presumably take
        you if you clicked on a link with an HREF as given.

        @param st: A relative URL, to be interpreted relative to C{self} as the
            base URL.
        @type st: L{bytes}

        @return: a new L{URLPath}
        """
    return self._fromURL(self._url.click(st.decode('ascii')))