from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.connection import Connection
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortios.fortios import (
from ansible_collections.fortinet.fortios.plugins.module_utils.fortimanager.common import (
from urllib.parse import quote
def fortios_monitor_fact(params, fos):
    valid, result = validate_parameters(params, fos)
    if not valid:
        return (True, False, result)
    selector = params['selector']
    url_params = dict()
    if params['filters'] and len(params['filters']):
        filter_body = quote(params['filters'][0])
        for filter_item in params['filters'][1:]:
            filter_body = '%s&filter=%s' % (filter_body, quote(filter_item))
        url_params['filter'] = filter_body
    if params['sorters'] and len(params['sorters']):
        sorter_body = params['sorters'][0]
        for sorter_item in params['sorters'][1:]:
            sorter_body = '%s&sort=%s' % (sorter_body, sorter_item)
        url_params['sort'] = sorter_body
    if params['formatters'] and len(params['formatters']):
        formatter_body = params['formatters'][0]
        for formatter_item in params['formatters'][1:]:
            formatter_body = '%s|%s' % (formatter_body, formatter_item)
        url_params['format'] = formatter_body
    if params['params']:
        for selector_param_key, selector_param in params['params'].items():
            url_params[selector_param_key] = selector_param
    fact = fos.monitor_get(module_selectors_defs[selector]['url'], params['vdom'], url_params)
    return (not is_successful_status(fact), False, fact)