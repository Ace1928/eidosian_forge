import datetime
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_response_header_policy(self, name, comment, cors_config, security_headers_config, custom_headers_config):
    cors_config = snake_dict_to_camel_dict(cors_config, capitalize_first=True)
    security_headers_config = snake_dict_to_camel_dict(security_headers_config, capitalize_first=True)
    if 'XssProtection' in security_headers_config:
        security_headers_config['XSSProtection'] = security_headers_config.pop('XssProtection')
    custom_headers_config = snake_dict_to_camel_dict(custom_headers_config, capitalize_first=True)
    config = {'Name': name, 'Comment': comment, 'CorsConfig': self.insert_quantities(cors_config), 'SecurityHeadersConfig': security_headers_config, 'CustomHeadersConfig': self.insert_quantities(custom_headers_config)}
    config = {k: v for k, v in config.items() if v}
    matching_policy = self.find_response_headers_policy(name)
    changed = False
    if self.check_mode:
        self.module.exit_json(changed=True, response_headers_policy=camel_dict_to_snake_dict(config))
    if matching_policy is None:
        try:
            result = self.client.create_response_headers_policy(ResponseHeadersPolicyConfig=config)
            changed = True
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Error creating policy')
    else:
        policy_id = matching_policy['ResponseHeadersPolicy']['Id']
        etag = matching_policy['ETag']
        try:
            result = self.client.update_response_headers_policy(Id=policy_id, IfMatch=etag, ResponseHeadersPolicyConfig=config)
            changed_time = result['ResponseHeadersPolicy']['LastModifiedTime']
            seconds = 3
            seconds_ago = datetime.datetime.now(changed_time.tzinfo) - datetime.timedelta(0, seconds)
            if changed_time > seconds_ago:
                changed = True
        except (ClientError, BotoCoreError) as e:
            self.module.fail_json_aws(e, msg='Updating creating policy')
    self.module.exit_json(changed=changed, **camel_dict_to_snake_dict(result))