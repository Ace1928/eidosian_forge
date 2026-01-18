import cherrypy
from cherrypy import _json as json
from cherrypy._cpcompat import text_or_bytes, ntou
def json_processor(entity):
    """Read application/json data into request.json."""
    if not entity.headers.get(ntou('Content-Length'), ntou('')):
        raise cherrypy.HTTPError(411)
    body = entity.fp.read()
    with cherrypy.HTTPError.handle(ValueError, 400, 'Invalid JSON document'):
        cherrypy.serving.request.json = json.decode(body.decode('utf-8'))