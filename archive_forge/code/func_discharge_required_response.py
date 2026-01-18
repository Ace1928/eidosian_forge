import json
from collections import namedtuple
import macaroonbakery.bakery as bakery
def discharge_required_response(macaroon, path, cookie_suffix_name, message=None):
    """ Get response content and headers from a discharge macaroons error.

    @param macaroon may hold a macaroon that, when discharged, may
    allow access to a service.
    @param path holds the URL path to be associated with the macaroon.
    The macaroon is potentially valid for all URLs under the given path.
    @param cookie_suffix_name holds the desired cookie name suffix to be
    associated with the macaroon. The actual name used will be
    ("macaroon-" + CookieName). Clients may ignore this field -
    older clients will always use ("macaroon-" + macaroon.signature() in hex)
    @return content(bytes) and the headers to set on the response(dict).
    """
    if message is None:
        message = 'discharge required'
    content = json.dumps({'Code': 'macaroon discharge required', 'Message': message, 'Info': {'Macaroon': macaroon.to_dict(), 'MacaroonPath': path, 'CookieNameSuffix': cookie_suffix_name}}).encode('utf-8')
    return (content, {'WWW-Authenticate': 'Macaroon', 'Content-Type': 'application/json'})