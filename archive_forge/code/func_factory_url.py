def factory_url(app):

    class URLForward(ForwardRequestExceptionMiddleware):

        def __call__(self, environ, start_response):
            environ['PATH_INFO'] = url.split('?')[0]
            environ['QUERY_STRING'] = url.split('?')[1]
            return self.app(environ, start_response)
    return URLForward(app)