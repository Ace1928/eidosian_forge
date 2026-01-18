from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_or_update_saml_provider(self, name, metadata):
    if not metadata:
        self.module.fail_json(msg='saml_metadata_document must be defined for present state')
    res = {'changed': False}
    try:
        arn = self._get_provider_arn(name)
    except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as e:
        self.module.fail_json_aws(e, msg=f"Could not get the ARN of the identity provider '{name}'")
    if arn:
        try:
            resp = self._get_saml_provider(arn)
        except (botocore.exceptions.ValidationError, botocore.exceptions.ClientError) as e:
            self.module.fail_json_aws(e, msg=f"Could not retrieve the identity provider '{name}'")
        if metadata.strip() != resp['SAMLMetadataDocument'].strip():
            res['changed'] = True
            if not self.module.check_mode:
                try:
                    resp = self._update_saml_provider(arn, metadata)
                    res['saml_provider'] = self._build_res(resp['SAMLProviderArn'])
                except botocore.exceptions.ClientError as e:
                    self.module.fail_json_aws(e, msg=f"Could not update the identity provider '{name}'")
        else:
            res['saml_provider'] = self._build_res(arn)
    else:
        res['changed'] = True
        if not self.module.check_mode:
            try:
                resp = self._create_saml_provider(metadata, name)
                res['saml_provider'] = self._build_res(resp['SAMLProviderArn'])
            except botocore.exceptions.ClientError as e:
                self.module.fail_json_aws(e, msg=f"Could not create the identity provider '{name}'")
    self.module.exit_json(**res)