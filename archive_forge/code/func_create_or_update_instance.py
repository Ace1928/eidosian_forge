import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_instance(module, client, instance_name):
    inst = find_instance_info(module, client, instance_name)
    if not inst:
        create_params = {'instanceNames': [instance_name], 'availabilityZone': module.params.get('zone'), 'blueprintId': module.params.get('blueprint_id'), 'bundleId': module.params.get('bundle_id'), 'userData': module.params.get('user_data')}
        key_pair_name = module.params.get('key_pair_name')
        if key_pair_name:
            create_params['keyPairName'] = key_pair_name
        try:
            client.create_instances(**create_params)
        except botocore.exceptions.ClientError as e:
            module.fail_json_aws(e)
        wait = module.params.get('wait')
        if wait:
            desired_states = ['running']
            wait_for_instance_state(module, client, instance_name, desired_states)
    if module.params.get('public_ports') is not None:
        update_public_ports(module, client, instance_name)
    after_update_inst = find_instance_info(module, client, instance_name, fail_if_not_found=True)
    module.exit_json(changed=after_update_inst != inst, instance=camel_dict_to_snake_dict(after_update_inst))