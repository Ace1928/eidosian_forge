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
def delete_and_wait(resource_type, client, get_fn, kwargs_get, delete_fn, kwargs_delete, module, states=None, wait_applicable=True, process_work_request=False):
    """A utility function to delete a resource and wait for the resource to get into the state as specified in the
    module options.
    :param wait_applicable: Specifies if wait for delete is applicable for this resource
    :param resource_type: Type of the resource to be deleted. e.g. "vcn"
    :param client: OCI service client instance to call the service periodically to retrieve data.
                   e.g. VirtualNetworkClient()
    :param get_fn: Function in the SDK to get the resource. e.g. virtual_network_client.get_vcn
    :param kwargs_get: Dictionary of arguments for get function get_fn. e.g. {"vcn_id": module.params["id"]}
    :param delete_fn: Function in the SDK to delete the resource. e.g. virtual_network_client.delete_vcn
    :param kwargs_delete: Dictionary of arguments for delete function delete_fn. e.g. {"vcn_id": module.params["id"]}
    :param module: Instance of AnsibleModule.
    :param states: List of lifecycle states to watch for while waiting after delete_fn is called. If nothing is passed,
                   defaults to ["TERMINATED", "DETACHED", "DELETED"].
    :param process_work_request: Whether a work request is generated on an API call and if it needs to be handled.
    :return: A dictionary containing the resource & the "changed" status. e.g. {"vcn":{x:y}, "changed":True}
    """
    states_set = set(['DETACHING', 'DETACHED', 'DELETING', 'DELETED', 'TERMINATING', 'TERMINATED'])
    result = dict(changed=False)
    result[resource_type] = dict()
    try:
        resource = to_dict(call_with_backoff(get_fn, **kwargs_get).data)
        if resource:
            if 'lifecycle_state' not in resource or resource['lifecycle_state'] not in states_set:
                response = call_with_backoff(delete_fn, **kwargs_delete)
                if process_work_request:
                    wr_id = response.headers.get('opc-work-request-id')
                    get_wr_response = call_with_backoff(client.get_work_request, work_request_id=wr_id)
                    result['work_request'] = to_dict(wait_on_work_request(client, get_wr_response, module))
                    result['changed'] = True
                    resource = to_dict(call_with_backoff(get_fn, **kwargs_get).data)
                else:
                    _debug('Deleted {0}, {1}'.format(resource_type, resource))
                    result['changed'] = True
                    if wait_applicable and module.params.get('wait', None):
                        if states is None:
                            states = module.params.get('wait_until') or DEFAULT_TERMINATED_STATES
                        try:
                            wait_response = oci.wait_until(client, get_fn(**kwargs_get), evaluate_response=lambda r: r.data.lifecycle_state in states, max_wait_seconds=module.params.get('wait_timeout', MAX_WAIT_TIMEOUT_IN_SECONDS), succeed_on_not_found=True)
                        except MaximumWaitTimeExceeded as ex:
                            module.fail_json(msg=str(ex))
                        except ServiceError as ex:
                            if ex.status != 404:
                                module.fail_json(msg=ex.message)
                            else:
                                _debug('API returned Status:404(Not Found) while waiting for resource to get into terminated state.')
                                resource['lifecycle_state'] = 'DELETED'
                                result[resource_type] = resource
                                return result
                        if not isinstance(wait_response, Sentinel):
                            resource = to_dict(wait_response.data)
                        else:
                            resource['lifecycle_state'] = 'DELETED'
            result[resource_type] = resource
        else:
            _debug('Resource {0} with {1} already deleted. So returning changed=False'.format(resource_type, kwargs_get))
    except ServiceError as ex:
        if isinstance(client, oci.dns.DnsClient):
            if ex.status == 400 and ex.code == 'InvalidParameter':
                _debug('Resource {0} with {1} already deleted. So returning changed=False'.format(resource_type, kwargs_get))
        elif ex.status != 404:
            module.fail_json(msg=ex.message)
        result[resource_type] = dict()
    return result