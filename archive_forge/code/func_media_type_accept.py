import weakref
import importlib_metadata
from wsme.exc import ClientSideError
def media_type_accept(request, content_types):
    """Validate media types against request.method.

    When request.method is GET or HEAD compare with the Accept header.
    When request.method is POST, PUT or PATCH compare with the Content-Type
    header.
    When request.method is DELETE media type is irrelevant, so return True.
    """
    if request.method in ['GET', 'HEAD']:
        if request.accept:
            if request.accept.acceptable_offers(content_types):
                return True
            error_message = 'Unacceptable Accept type: %s not in %s' % (request.accept, content_types)
            raise ClientSideError(error_message, status_code=406)
    elif request.method in ['PUT', 'POST', 'PATCH']:
        content_type = request.headers.get('Content-Type')
        if content_type:
            for ct in content_types:
                if request.headers.get('Content-Type', '').startswith(ct):
                    return True
            error_message = 'Unacceptable Content-Type: %s not in %s' % (content_type, content_types)
            raise ClientSideError(error_message, status_code=415)
        else:
            raise ClientSideError('missing Content-Type header')
    elif request.method in ['DELETE']:
        return True
    return False