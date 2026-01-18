import functools
import json
import jsonschema
import yaml
from ironicclient import exc
def create_single_handler(resource_type):
    """Catch errors of the creation of a single resource.

    This decorator appends an error (which is an instance of some client
    exception class) to the return value of the create_method, changing the
    return value from just UUID to (UUID, error), and does some exception
    handling.

    :param resource_type: string value, the type of the resource being created,
        e.g. 'node', used purely for exception messages.
    """

    def outer_wrapper(create_method):

        @functools.wraps(create_method)
        def wrapper(client, **params):
            uuid = None
            error = None
            try:
                uuid = create_method(client, **params)
            except exc.InvalidAttribute as e:
                error = exc.InvalidAttribute('Cannot create the %(resource)s with attributes %(params)s. One or more attributes are invalid: %(err)s' % {'params': params, 'resource': resource_type, 'err': e})
            except Exception as e:
                error = exc.ClientException('Unable to create the %(resource)s with the specified attributes: %(params)s. The error is: %(error)s' % {'error': e, 'resource': resource_type, 'params': params})
            return (uuid, error)
        return wrapper
    return outer_wrapper