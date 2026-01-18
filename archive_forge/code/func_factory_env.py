def factory_env(app):

    class EnvironForward(ForwardRequestExceptionMiddleware):

        def __call__(self, environ_, start_response):
            return self.app(environ, start_response)
    return EnvironForward(app)