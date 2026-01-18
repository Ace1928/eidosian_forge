import cherrypy
from cherrypy import _json as json
from cherrypy._cpcompat import text_or_bytes, ntou
def json_out(content_type='application/json', debug=False, handler=json_handler):
    """Wrap request.handler to serialize its output to JSON. Sets Content-Type.

    If the given content_type is None, the Content-Type response header
    is not set.

    Provide your own handler to use a custom encoder.  For example
    cherrypy.config['tools.json_out.handler'] = <function>, or
    @json_out(handler=function).
    """
    request = cherrypy.serving.request
    if request.handler is None:
        return
    if debug:
        cherrypy.log('Replacing %s with JSON handler' % request.handler, 'TOOLS.JSON_OUT')
    request._json_inner_handler = request.handler
    request.handler = handler
    if content_type is not None:
        if debug:
            cherrypy.log('Setting Content-Type to %s' % content_type, 'TOOLS.JSON_OUT')
        cherrypy.serving.response.headers['Content-Type'] = content_type