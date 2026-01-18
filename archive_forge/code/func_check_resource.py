import functools
from .. import errors
from . import utils
def check_resource(resource_name):

    def decorator(f):

        @functools.wraps(f)
        def wrapped(self, resource_id=None, *args, **kwargs):
            if resource_id is None and kwargs.get(resource_name):
                resource_id = kwargs.pop(resource_name)
            if isinstance(resource_id, dict):
                resource_id = resource_id.get('Id', resource_id.get('ID'))
            if not resource_id:
                raise errors.NullResource('Resource ID was not provided')
            return f(self, resource_id, *args, **kwargs)
        return wrapped
    return decorator