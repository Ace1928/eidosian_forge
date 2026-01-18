from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def check_and_update_resource(resource_type, get_fn, kwargs_get, update_fn, primitive_params_update, kwargs_non_primitive_update, module, update_attributes, client=None, sub_attributes_of_update_model=None, wait_applicable=True, states=None):
    """
    This function handles update operation on a resource. It checks whether update is required and accordingly returns
    the resource and the changed status.
    :param wait_applicable: Indicates if the resource support wait
    :param client:  The resource Client class to use to perform the wait checks. This param must be specified if
            wait_applicable is True
    :param resource_type: The type of the resource. e.g. "private_ip"
    :param get_fn: Function used to get the resource. e.g. virtual_network_client.get_private_ip
    :param kwargs_get: Dictionary containing the arguments to be used to call get function.
           e.g. {"private_ip_id": module.params["private_ip_id"]}
    :param update_fn: Function used to update the resource. e.g virtual_network_client.update_private_ip
    :param primitive_params_update: List of primitive parameters used for update function. e.g. ['private_ip_id']
    :param kwargs_non_primitive_update: Dictionary containing the non-primitive arguments to be used to call get
     function with key as the non-primitive argument type & value as the name of the non-primitive argument to be passed
     to the update function. e.g. {UpdatePrivateIpDetails: "update_private_ip_details"}
    :param module: Instance of AnsibleModule
    :param update_attributes: Attributes in update model.
    :param states: List of lifecycle states to watch for while waiting after create_fn is called.
                   e.g. [module.params['wait_until'], "FAULTY"]
    :param sub_attributes_of_update_model: Dictionary of non-primitive sub-attributes of update model. for example,
        {'services': [ServiceIdRequestDetails()]} as in UpdateServiceGatewayDetails.
    :return: Returns a dictionary containing the "changed" status and the resource.
    """
    try:
        result = dict(changed=False)
        attributes_to_update, resource = get_attr_to_update(get_fn, kwargs_get, module, update_attributes)
        if attributes_to_update:
            kwargs_update = get_kwargs_update(attributes_to_update, kwargs_non_primitive_update, module, primitive_params_update, sub_attributes_of_update_model)
            resource = call_with_backoff(update_fn, **kwargs_update).data
            if wait_applicable:
                if client is None:
                    module.fail_json(msg='wait_applicable is True, but client is not specified.')
                resource = wait_for_resource_lifecycle_state(client, module, True, kwargs_get, get_fn, None, resource, states)
            result['changed'] = True
        result[resource_type] = to_dict(resource)
        return result
    except ServiceError as ex:
        module.fail_json(msg=ex.message)