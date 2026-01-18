from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def record_sets_details():
    params = dict()
    if module.params.get('hosted_zone_id'):
        params['HostedZoneId'] = module.params.get('hosted_zone_id')
    else:
        module.fail_json(msg='Hosted Zone Id is required')
    if module.params.get('start_record_name'):
        params['StartRecordName'] = module.params.get('start_record_name')
    if module.params.get('type') and (not module.params.get('start_record_name')):
        module.fail_json(msg='start_record_name must be specified if type is set')
    if module.params.get('type'):
        params['StartRecordType'] = module.params.get('type')
    if module.params.get('max_items'):
        params['PaginationConfig'] = dict(MaxItems=module.params.get('max_items'))
    record_sets = _paginated_result('list_resource_record_sets', **params)['ResourceRecordSets']
    snaked_record_sets = [camel_dict_to_snake_dict(record_set) for record_set in record_sets]
    module.deprecate("The 'CamelCase' return values with key 'ResourceRecordSets' and 'list' are deprecated and will be replaced by 'snake_case' return values with key 'resource_record_sets'.  Both case values are returned for now.", date='2025-01-01', collection_name='amazon.aws')
    return {'ResourceRecordSets': record_sets, 'list': record_sets, 'resource_record_sets': snaked_record_sets}