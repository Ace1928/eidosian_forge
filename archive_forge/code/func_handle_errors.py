from collections import namedtuple
from time import sleep
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .tagging import compare_aws_tags
from .waiters import get_waiter
def handle_errors(module, exception, method_name, parameters):
    if not isinstance(exception, ClientError):
        module.fail_json_aws(exception, msg=f'Unexpected failure for method {method_name} with parameters {parameters}')
    changed = True
    error_code = exception.response['Error']['Code']
    if method_name in ('modify_db_instance', 'modify_db_cluster') and error_code == 'InvalidParameterCombination':
        if 'No modifications were requested' in to_text(exception):
            changed = False
        elif 'ModifyDbCluster API' in to_text(exception):
            module.fail_json_aws(exception, msg='It appears you are trying to modify attributes that are managed at the cluster level. Please see rds_cluster')
        else:
            module.fail_json_aws(exception, msg=f'Unable to {get_rds_method_attribute(method_name, module).operation_description}')
    elif method_name == 'promote_read_replica' and error_code == 'InvalidDBInstanceState':
        if 'DB Instance is not a read replica' in to_text(exception):
            changed = False
        else:
            module.fail_json_aws(exception, msg=f'Unable to {get_rds_method_attribute(method_name, module).operation_description}')
    elif method_name == 'promote_read_replica_db_cluster' and error_code == 'InvalidDBClusterStateFault':
        if 'DB Cluster that is not a read replica' in to_text(exception):
            changed = False
        else:
            module.fail_json_aws(exception, msg=f'Unable to {get_rds_method_attribute(method_name, module).operation_description}')
    elif method_name == 'create_db_cluster' and error_code == 'InvalidParameterValue':
        accepted_engines = ['aurora', 'aurora-mysql', 'aurora-postgresql', 'mysql', 'postgres']
        if parameters.get('Engine') not in accepted_engines:
            module.fail_json_aws(exception, msg=f'DB engine {parameters.get('Engine')} should be one of {accepted_engines}')
        else:
            module.fail_json_aws(exception, msg=f'Unable to {get_rds_method_attribute(method_name, module).operation_description}')
    else:
        module.fail_json_aws(exception, msg=f'Unable to {get_rds_method_attribute(method_name, module).operation_description}')
    return changed