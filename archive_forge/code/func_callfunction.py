import functools
import inspect
import sys
import pecan
import wsme
import wsme.rest.args
import wsme.rest.json
import wsme.rest.xml
from wsme.utils import is_valid_code
@functools.wraps(f)
def callfunction(self, *args, **kwargs):
    return_type = funcdef.return_type
    try:
        args, kwargs = wsme.rest.args.get_args(funcdef, args, kwargs, pecan.request.params, None, pecan.request.body, pecan.request.content_type)
        if funcdef.pass_request:
            kwargs[funcdef.pass_request] = pecan.request
        result = f(self, *args, **kwargs)
        pecan.response.status = funcdef.status_code
        if isinstance(result, wsme.api.Response):
            pecan.response.status = result.status_code
            if result.status_code == 204:
                return_type = None
            elif not isinstance(result.return_type, wsme.types.UnsetType):
                return_type = result.return_type
            result = result.obj
    except Exception:
        try:
            exception_info = sys.exc_info()
            orig_exception = exception_info[1]
            orig_code = getattr(orig_exception, 'code', None)
            data = wsme.api.format_exception(exception_info, pecan.conf.get('wsme', {}).get('debug', False))
        finally:
            del exception_info
        if orig_code and is_valid_code(orig_code):
            pecan.response.status = orig_code
        else:
            pecan.response.status = 500
        return data
    if return_type is None:
        pecan.request.pecan['content_type'] = None
        pecan.response.content_type = None
        return ''
    return dict(datatype=return_type, result=result)