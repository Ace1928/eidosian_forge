from __future__ import absolute_import, division, print_function
import copy
from functools import partial
from ansible_collections.community.network.plugins.module_utils.network.ftd.common import HTTPMethod, equal_objects, FtdConfigurationError, \
from ansible_collections.community.network.plugins.module_utils.network.ftd.fdm_swagger_client import OperationField, ValidationError
from ansible.module_utils.six import iteritems
def iterate_over_pageable_resource(resource_func, params):
    """
    A generator function that iterates over a resource that supports pagination and lazily returns present items
    one by one.

    :param resource_func: function that receives `params` argument and returns a page of objects
    :type resource_func: callable
    :param params: initial dictionary of parameters that will be passed to the resource_func.
                   Should contain `query_params` inside.
    :type params: dict
    :return: an iterator containing returned items
    :rtype: iterator of dict
    """
    params = copy.deepcopy(params)
    params[ParamName.QUERY_PARAMS].setdefault('limit', DEFAULT_PAGE_SIZE)
    params[ParamName.QUERY_PARAMS].setdefault('offset', DEFAULT_OFFSET)
    limit = int(params[ParamName.QUERY_PARAMS]['limit'])

    def received_less_items_than_requested(items_in_response, items_expected):
        if items_in_response == items_expected:
            return False
        elif items_in_response < items_expected:
            return True
        raise FtdUnexpectedResponse('Get List of Objects Response from the server contains more objects than requested. There are {0} item(s) in the response while {1} was(ere) requested'.format(items_in_response, items_expected))
    while True:
        result = resource_func(params=params)
        for item in result['items']:
            yield item
        if received_less_items_than_requested(len(result['items']), limit):
            break
        params = copy.deepcopy(params)
        query_params = params[ParamName.QUERY_PARAMS]
        query_params['offset'] = int(query_params['offset']) + limit