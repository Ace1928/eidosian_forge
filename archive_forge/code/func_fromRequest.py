from typing import cast
from urllib.parse import quote as urlquote, unquote as urlunquote, urlunsplit
from hyperlink import URL as _URL
@classmethod
def fromRequest(klass, request):
    """
        Make a L{URLPath} from a L{twisted.web.http.Request}.

        @param request: A L{twisted.web.http.Request} to make the L{URLPath}
            from.

        @return: a new L{URLPath} derived from the given request.
        @rtype: L{URLPath}
        """
    return klass.fromBytes(request.prePathURL())