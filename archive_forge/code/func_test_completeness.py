from __future__ import absolute_import, division, print_function
from awx.main.tests.functional.conftest import _request
from ansible.module_utils.six import string_types
import yaml
import os
import re
import glob
def test_completeness(collection_import, request, admin_user, job_template, execution_environment):
    option_comparison = {}
    base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    module_directory = os.path.join(base_folder, 'plugins', 'modules')
    for root, dirs, files in os.walk(module_directory):
        if root == module_directory:
            for filename in files:
                if os.path.islink(os.path.join(root, filename)):
                    continue
                if re.match('^[a-z].*.py$', filename):
                    module_name = filename[:-3]
                    option_comparison[module_name] = {'endpoint': 'N/A', 'api_options': {}, 'module_options': {}, 'module_name': module_name}
                    resource_module = collection_import('plugins.modules.{0}'.format(module_name))
                    option_comparison[module_name]['module_options'] = yaml.load(resource_module.DOCUMENTATION, Loader=yaml.SafeLoader)['options']
    endpoint_response = _request('get')(url='/api/v2/', user=admin_user, expect=None)
    for key, val in extra_endpoints.items():
        endpoint_response.data[key] = val
    for endpoint in endpoint_response.data.keys():
        singular_endpoint = '{0}'.format(endpoint)
        if singular_endpoint.endswith('ies'):
            singular_endpoint = singular_endpoint[:-3]
        elif singular_endpoint != 'settings' and singular_endpoint.endswith('s'):
            singular_endpoint = singular_endpoint[:-1]
        module_name = '{0}'.format(singular_endpoint)
        endpoint_url = endpoint_response.data.get(endpoint)
        if module_name not in option_comparison:
            option_comparison[module_name] = {}
            option_comparison[module_name]['module_name'] = 'N/A'
            option_comparison[module_name]['module_options'] = {}
        option_comparison[module_name]['endpoint'] = endpoint_url
        option_comparison[module_name]['api_options'] = {}
        options_response = _request('options')(url=endpoint_url, user=admin_user, expect=None)
        if 'POST' in options_response.data.get('actions', {}):
            option_comparison[module_name]['api_options'] = options_response.data.get('actions').get('POST')
        else:
            read_only_endpoint.append(module_name)
    longest_module_name = 0
    longest_option_name = 0
    longest_endpoint = 0
    for module, module_value in option_comparison.items():
        if len(module_value['module_name']) > longest_module_name:
            longest_module_name = len(module_value['module_name'])
        if len(module_value['endpoint']) > longest_endpoint:
            longest_endpoint = len(module_value['endpoint'])
        for option in (module_value['api_options'], module_value['module_options']):
            if len(option) > longest_option_name:
                longest_option_name = len(option)
    print(''.join(['End Point', ' ' * (longest_endpoint - len('End Point')), ' | Module Name', ' ' * (longest_module_name - len('Module Name')), ' | Option', ' ' * (longest_option_name - len('Option')), ' | API | Module | State']))
    print('-|-'.join(['-' * longest_endpoint, '-' * longest_module_name, '-' * longest_option_name, '---', '------', '---------------------------------------------']))
    for module in sorted(option_comparison):
        module_data = option_comparison[module]
        all_param_names = list(set(module_data['api_options']) | set(module_data['module_options']))
        for parameter in sorted(all_param_names):
            print(''.join([module_data['endpoint'], ' ' * (longest_endpoint - len(module_data['endpoint'])), ' | ', module_data['module_name'], ' ' * (longest_module_name - len(module_data['module_name'])), ' | ', parameter, ' ' * (longest_option_name - len(parameter)), ' | ', ' X ' if parameter in module_data['api_options'] else '   ', ' | ', '  X   ' if parameter in module_data['module_options'] else '      ', ' | ', determine_state(module, module_data['endpoint'], module_data['module_name'], parameter, module_data['api_options'][parameter] if parameter in module_data['api_options'] else None, module_data['module_options'][parameter] if parameter in module_data['module_options'] else None)]))
        if len(all_param_names) == 0:
            print(''.join([module_data['endpoint'], ' ' * (longest_endpoint - len(module_data['endpoint'])), ' | ', module_data['module_name'], ' ' * (longest_module_name - len(module_data['module_name'])), ' | ', 'N/A', ' ' * (longest_option_name - len('N/A')), ' | ', '   ', ' | ', '      ', ' | ', determine_state(module, module_data['endpoint'], module_data['module_name'], 'N/A', None, None)]))
    test_meta_runtime()
    if return_value != 0:
        raise Exception('One or more failures caused issues')