from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import get_boto3_client_method_parameters
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.rds import arg_spec_to_rds_params
from ansible_collections.amazon.aws.plugins.module_utils.rds import call_method
from ansible_collections.amazon.aws.plugins.module_utils.rds import ensure_tags
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_rds_method_attribute
from ansible_collections.amazon.aws.plugins.module_utils.rds import get_tags
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
def ensure_snapshot_absent():
    snapshot_name = module.params.get('db_snapshot_identifier')
    params = {'DBSnapshotIdentifier': snapshot_name}
    changed = False
    snapshot = get_snapshot(snapshot_name)
    if not snapshot:
        module.exit_json(changed=changed)
    elif snapshot and snapshot['Status'] != 'deleting':
        snapshot, changed = call_method(client, module, 'delete_db_snapshot', params)
    module.exit_json(changed=changed)