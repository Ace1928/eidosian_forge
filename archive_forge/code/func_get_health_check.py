from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_health_check():
    params = dict()
    results = dict()
    if not module.params.get('health_check_id'):
        module.fail_json(msg='health_check_id is required')
    else:
        params['HealthCheckId'] = module.params.get('health_check_id')
    if module.params.get('health_check_method') == 'details':
        results = client.get_health_check(**params)
        results['health_check'] = camel_dict_to_snake_dict(results['HealthCheck'])
        module.deprecate("The 'CamelCase' return values with key 'HealthCheck' is deprecated and will be replaced by 'snake_case' return values with key 'health_check'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    elif module.params.get('health_check_method') == 'failure_reason':
        response = client.get_health_check_last_failure_reason(**params)
        results['health_check_observations'] = [camel_dict_to_snake_dict(health_check) for health_check in response['HealthCheckObservations']]
    elif module.params.get('health_check_method') == 'status':
        response = client.get_health_check_status(**params)
        results['health_check_observations'] = [camel_dict_to_snake_dict(health_check) for health_check in response['HealthCheckObservations']]
    return results