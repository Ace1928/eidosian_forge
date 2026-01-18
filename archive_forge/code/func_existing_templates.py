from uuid import uuid4
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.arn import validate_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def existing_templates(module):
    ec2 = module.client('ec2', retry_decorator=AWSRetry.jittered_backoff())
    matches = None
    try:
        if module.params.get('template_id'):
            matches = ec2.describe_launch_templates(LaunchTemplateIds=[module.params.get('template_id')], aws_retry=True)
        elif module.params.get('template_name'):
            matches = ec2.describe_launch_templates(LaunchTemplateNames=[module.params.get('template_name')], aws_retry=True)
    except is_boto3_error_code('InvalidLaunchTemplateName.NotFoundException') as e:
        return (None, [])
    except is_boto3_error_code('InvalidLaunchTemplateId.Malformed') as e:
        module.fail_json_aws(e, msg=f'Launch template with ID {module.params.get('launch_template_id')} is not a valid ID. It should start with `lt-....`')
    except is_boto3_error_code('InvalidLaunchTemplateId.NotFoundException') as e:
        module.fail_json_aws(e, msg=f'Launch template with ID {module.params.get('launch_template_id')} could not be found, please supply a name instead so that a new template can be created')
    except (ClientError, BotoCoreError, WaiterError) as e:
        module.fail_json_aws(e, msg='Could not check existing launch templates. This may be an IAM permission problem.')
    else:
        template = matches['LaunchTemplates'][0]
        template_id, template_version, template_default = (template['LaunchTemplateId'], template['LatestVersionNumber'], template['DefaultVersionNumber'])
        try:
            return (template, ec2.describe_launch_template_versions(LaunchTemplateId=template_id, aws_retry=True)['LaunchTemplateVersions'])
        except (ClientError, BotoCoreError, WaiterError) as e:
            module.fail_json_aws(e, msg=f'Could not find launch template versions for {template['LaunchTemplateName']} (ID: {template_id}).')