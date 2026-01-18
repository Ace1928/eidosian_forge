from paste import httpexceptions
from paste import wsgilib
import flup.middleware.session
def cookie_start_response(status, headers, exc_info=None):
    service.addCookie(headers)
    return start_response(status, headers, exc_info)