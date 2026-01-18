from functools import partial, update_wrapper, wraps
from asgiref.sync import iscoroutinefunction
def make_middleware_decorator(middleware_class):

    def _make_decorator(*m_args, **m_kwargs):

        def _decorator(view_func):
            middleware = middleware_class(view_func, *m_args, **m_kwargs)

            def _pre_process_request(request, *args, **kwargs):
                if hasattr(middleware, 'process_request'):
                    result = middleware.process_request(request)
                    if result is not None:
                        return result
                if hasattr(middleware, 'process_view'):
                    result = middleware.process_view(request, view_func, args, kwargs)
                    if result is not None:
                        return result
                return None

            def _process_exception(request, exception):
                if hasattr(middleware, 'process_exception'):
                    result = middleware.process_exception(request, exception)
                    if result is not None:
                        return result
                raise

            def _post_process_request(request, response):
                if hasattr(response, 'render') and callable(response.render):
                    if hasattr(middleware, 'process_template_response'):
                        response = middleware.process_template_response(request, response)
                    if hasattr(middleware, 'process_response'):

                        def callback(response):
                            return middleware.process_response(request, response)
                        response.add_post_render_callback(callback)
                elif hasattr(middleware, 'process_response'):
                    return middleware.process_response(request, response)
                return response
            if iscoroutinefunction(view_func):

                async def _view_wrapper(request, *args, **kwargs):
                    result = _pre_process_request(request, *args, **kwargs)
                    if result is not None:
                        return result
                    try:
                        response = await view_func(request, *args, **kwargs)
                    except Exception as e:
                        result = _process_exception(request, e)
                        if result is not None:
                            return result
                    return _post_process_request(request, response)
            else:

                def _view_wrapper(request, *args, **kwargs):
                    result = _pre_process_request(request, *args, **kwargs)
                    if result is not None:
                        return result
                    try:
                        response = view_func(request, *args, **kwargs)
                    except Exception as e:
                        result = _process_exception(request, e)
                        if result is not None:
                            return result
                    return _post_process_request(request, response)
            return wraps(view_func)(_view_wrapper)
        return _decorator
    return _make_decorator