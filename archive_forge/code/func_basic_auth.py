import binascii
import unicodedata
import base64
import cherrypy
from cherrypy._cpcompat import ntou, tonative
def basic_auth(realm, checkpassword, debug=False, accept_charset='utf-8'):
    """A CherryPy tool which hooks at before_handler to perform
    HTTP Basic Access Authentication, as specified in :rfc:`2617`
    and :rfc:`7617`.

    If the request has an 'authorization' header with a 'Basic' scheme, this
    tool attempts to authenticate the credentials supplied in that header.  If
    the request has no 'authorization' header, or if it does but the scheme is
    not 'Basic', or if authentication fails, the tool sends a 401 response with
    a 'WWW-Authenticate' Basic header.

    realm
        A string containing the authentication realm.

    checkpassword
        A callable which checks the authentication credentials.
        Its signature is checkpassword(realm, username, password). where
        username and password are the values obtained from the request's
        'authorization' header.  If authentication succeeds, checkpassword
        returns True, else it returns False.

    """
    fallback_charset = 'ISO-8859-1'
    if '"' in realm:
        raise ValueError('Realm cannot contain the " (quote) character.')
    request = cherrypy.serving.request
    auth_header = request.headers.get('authorization')
    if auth_header is not None:
        msg = 'Bad Request'
        with cherrypy.HTTPError.handle((ValueError, binascii.Error), 400, msg):
            scheme, params = auth_header.split(' ', 1)
            if scheme.lower() == 'basic':
                charsets = (accept_charset, fallback_charset)
                decoded_params = base64.b64decode(params.encode('ascii'))
                decoded_params = _try_decode(decoded_params, charsets)
                decoded_params = ntou(decoded_params)
                decoded_params = unicodedata.normalize('NFC', decoded_params)
                decoded_params = tonative(decoded_params)
                username, password = decoded_params.split(':', 1)
                if checkpassword(realm, username, password):
                    if debug:
                        cherrypy.log('Auth succeeded', 'TOOLS.AUTH_BASIC')
                    request.login = username
                    return
    charset = accept_charset.upper()
    charset_declaration = ', charset="%s"' % charset if charset != fallback_charset else ''
    cherrypy.serving.response.headers['www-authenticate'] = 'Basic realm="%s"%s' % (realm, charset_declaration)
    raise cherrypy.HTTPError(401, 'You are not authorized to access that resource')