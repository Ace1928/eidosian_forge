import sys
import time
from scgi import scgi_server
def serve_application(application, prefix, port=None, host=None, max_children=None):
    """
    Serve the specified WSGI application via SCGI proxy.

    ``application``
        The WSGI application to serve.

    ``prefix``
        The prefix for what is served by the SCGI Web-server-side process.

    ``port``
        Optional port to bind the SCGI proxy to. Defaults to SCGIServer's
        default port value.

    ``host``
        Optional host to bind the SCGI proxy to. Defaults to SCGIServer's
        default host value.

    ``host``
        Optional maximum number of child processes the SCGIServer will
        spawn. Defaults to SCGIServer's default max_children value.
    """

    class SCGIAppHandler(SWAP):

        def __init__(self, *args, **kwargs):
            self.prefix = prefix
            self.app_obj = application
            SWAP.__init__(self, *args, **kwargs)
    kwargs = dict(handler_class=SCGIAppHandler)
    for kwarg in ('host', 'port', 'max_children'):
        if locals()[kwarg] is not None:
            kwargs[kwarg] = locals()[kwarg]
    scgi_server.SCGIServer(**kwargs).serve()