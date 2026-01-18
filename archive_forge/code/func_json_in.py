import cherrypy
from cherrypy import _json as json
from cherrypy._cpcompat import text_or_bytes, ntou
def json_in(content_type=[ntou('application/json'), ntou('text/javascript')], force=True, debug=False, processor=json_processor):
    """Add a processor to parse JSON request entities:
    The default processor places the parsed data into request.json.

    Incoming request entities which match the given content_type(s) will
    be deserialized from JSON to the Python equivalent, and the result
    stored at cherrypy.request.json. The 'content_type' argument may
    be a Content-Type string or a list of allowable Content-Type strings.

    If the 'force' argument is True (the default), then entities of other
    content types will not be allowed; "415 Unsupported Media Type" is
    raised instead.

    Supply your own processor to use a custom decoder, or to handle the parsed
    data differently.  The processor can be configured via
    tools.json_in.processor or via the decorator method.

    Note that the deserializer requires the client send a Content-Length
    request header, or it will raise "411 Length Required". If for any
    other reason the request entity cannot be deserialized from JSON,
    it will raise "400 Bad Request: Invalid JSON document".
    """
    request = cherrypy.serving.request
    if isinstance(content_type, text_or_bytes):
        content_type = [content_type]
    if force:
        if debug:
            cherrypy.log('Removing body processors %s' % repr(request.body.processors.keys()), 'TOOLS.JSON_IN')
        request.body.processors.clear()
        request.body.default_proc = cherrypy.HTTPError(415, 'Expected an entity of content type %s' % ', '.join(content_type))
    for ct in content_type:
        if debug:
            cherrypy.log('Adding body processor for %s' % ct, 'TOOLS.JSON_IN')
        request.body.processors[ct] = processor