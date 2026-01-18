import time
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def enable_or_update_bucket_as_website(client_connection, resource_connection, module):
    bucket_name = module.params.get('name')
    redirect_all_requests = module.params.get('redirect_all_requests')
    if redirect_all_requests is not None:
        suffix = None
    else:
        suffix = module.params.get('suffix')
    error_key = module.params.get('error_key')
    changed = False
    try:
        bucket_website = resource_connection.BucketWebsite(bucket_name)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get bucket')
    try:
        website_config = client_connection.get_bucket_website(Bucket=bucket_name)
    except is_boto3_error_code('NoSuchWebsiteConfiguration'):
        website_config = None
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        module.fail_json_aws(e, msg='Failed to get website configuration')
    if website_config is None:
        try:
            bucket_website.put(WebsiteConfiguration=_create_website_configuration(suffix, error_key, redirect_all_requests))
            changed = True
        except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
            module.fail_json_aws(e, msg='Failed to set bucket website configuration')
        except ValueError as e:
            module.fail_json(msg=str(e))
    else:
        try:
            if suffix is not None and website_config['IndexDocument']['Suffix'] != suffix or (error_key is not None and website_config['ErrorDocument']['Key'] != error_key) or (redirect_all_requests is not None and website_config['RedirectAllRequestsTo'] != _create_redirect_dict(redirect_all_requests)):
                try:
                    bucket_website.put(WebsiteConfiguration=_create_website_configuration(suffix, error_key, redirect_all_requests))
                    changed = True
                except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                    module.fail_json_aws(e, msg='Failed to update bucket website configuration')
        except KeyError as e:
            try:
                bucket_website.put(WebsiteConfiguration=_create_website_configuration(suffix, error_key, redirect_all_requests))
                changed = True
            except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
                module.fail_json_aws(e, msg='Failed to update bucket website configuration')
        except ValueError as e:
            module.fail_json(msg=str(e))
        time.sleep(5)
    website_config = client_connection.get_bucket_website(Bucket=bucket_name)
    module.exit_json(changed=changed, **camel_dict_to_snake_dict(website_config))