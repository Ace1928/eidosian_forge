from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def normalize_credentials(credentials):
    access_key = credentials.get('AccessKeyId', None)
    secret_key = credentials.get('SecretAccessKey', None)
    session_token = credentials.get('SessionToken', None)
    expiration = credentials.get('Expiration', None)
    return {'access_key': access_key, 'secret_key': secret_key, 'session_token': session_token, 'expiration': expiration}