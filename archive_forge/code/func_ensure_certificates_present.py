import base64
import re  # regex library
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible_collections.amazon.aws.plugins.module_utils.acm import ACMServiceManager
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def ensure_certificates_present(client, module, acm, certificates, desired_tags, filter_tags):
    cert_arn = None
    changed = False
    if len(certificates) > 1:
        msg = f'More than one certificate with Name={module.params['name_tag']} exists in ACM in this region'
        module.fail_json(msg=msg, certificates=certificates)
    elif len(certificates) == 1:
        changed, cert_arn = update_imported_certificate(client, module, acm, certificates[0], desired_tags)
    else:
        changed, cert_arn = import_certificate(client, module, acm, desired_tags)
    try:
        existing_tags = boto3_tag_list_to_ansible_dict(client.list_tags_for_certificate(CertificateArn=cert_arn)['Tags'])
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, "Couldn't get tags for certificate")
    purge_tags = module.params.get('purge_tags')
    c, new_tags = ensure_tags(client, module, cert_arn, existing_tags, desired_tags, purge_tags)
    changed |= c
    domain = acm.get_domain_of_cert(client=client, module=module, arn=cert_arn)
    module.exit_json(certificate=dict(domain_name=domain, arn=cert_arn, tags=new_tags), changed=changed)