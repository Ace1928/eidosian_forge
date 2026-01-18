import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
def catch_401_app_call(self, environ, start_response):
    """
        Call the application, and redirect if the app returns a 401 response
        """
    was_401 = []

    def replacement_start_response(status, headers, exc_info=None):
        if int(status.split(None, 1)) == 401:
            was_401.append(1)

            def dummy_writer(v):
                pass
            return dummy_writer
        else:
            return start_response(status, headers, exc_info)
    app_iter = self.app(environ, replacement_start_response)
    if was_401:
        try:
            list(app_iter)
        finally:
            if hasattr(app_iter, 'close'):
                app_iter.close()
        redir_url = paste.request.construct_url(environ, with_path_info=False, with_query_string=False)
        exc = httpexceptions.HTTPTemporaryRedirect(redir_url)
        return exc.wsgi_application(environ, start_response)
    else:
        return app_iter