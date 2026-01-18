import os
import boto
from boto.compat import json
from boto.exception import BotoClientError
from boto.endpoints import BotoEndpointResolver
from boto.endpoints import StaticEndpointBuilder
def merge_endpoints(defaults, additions):
    """
    Given an existing set of endpoint data, this will deep-update it with
    any similarly structured data in the additions.

    :param defaults: The existing endpoints data
    :type defaults: dict

    :param defaults: The additional endpoints data
    :type defaults: dict

    :returns: The modified endpoints data
    :rtype: dict
    """
    for service, region_info in additions.items():
        defaults.setdefault(service, {})
        defaults[service].update(region_info)
    return defaults