from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def list_health_checks():
    params = dict()
    if module.params.get('next_marker'):
        params['Marker'] = module.params.get('next_marker')
    if module.params.get('max_items'):
        params['PaginationConfig'] = dict(MaxItems=module.params.get('max_items'))
    health_checks = _paginated_result('list_health_checks', **params)['HealthChecks']
    snaked_health_checks = [camel_dict_to_snake_dict(health_check) for health_check in health_checks]
    module.deprecate("The 'CamelCase' return values with key 'HealthChecks' and 'list' are deprecated and will be replaced by 'snake_case' return values with key 'health_checks'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    return {'HealthChecks': health_checks, 'list': health_checks, 'health_checks': snaked_health_checks}