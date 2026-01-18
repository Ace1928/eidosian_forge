import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def load_endpoint_json(path):
    """
    Loads a given JSON file & returns it.

    :param path: The path to the JSON file
    :type path: string

    :returns: The loaded data
    """
    return _load_json_file(path)