import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def load_regions():
    """
    Actually load the region/endpoint information from the JSON files.

    By default, this loads from the default included ``boto/endpoints.json``
    file.

    Users can override/extend this by supplying either a ``BOTO_ENDPOINTS``
    environment variable or a ``endpoints_path`` config variable, either of
    which should be an absolute path to the user's JSON file.

    :returns: The endpoints data
    :rtype: dict
    """
    endpoints = _load_builtin_endpoints()
    additional_path = None
    if os.environ.get('BOTO_ENDPOINTS'):
        additional_path = os.environ['BOTO_ENDPOINTS']
    elif boto.config.get('Boto', 'endpoints_path'):
        additional_path = boto.config.get('Boto', 'endpoints_path')
    if additional_path:
        additional = load_endpoint_json(additional_path)
        endpoints = merge_endpoints(endpoints, additional)
    return endpoints