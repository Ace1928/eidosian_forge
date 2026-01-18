from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def create_layer_version(lambda_client, params, check_mode=False):
    if check_mode:
        return {'msg': 'Create operation skipped - running in check mode', 'changed': True}
    opt = {'LayerName': params.get('name'), 'Content': {}}
    keys = [('description', 'Description'), ('compatible_runtimes', 'CompatibleRuntimes'), ('license_info', 'LicenseInfo'), ('compatible_architectures', 'CompatibleArchitectures')]
    for k, d in keys:
        if params.get(k) is not None:
            opt[d] = params.get(k)
    zip_file = params['content'].get('zip_file')
    if zip_file is not None:
        with open(zip_file, 'rb') as zf:
            opt['Content']['ZipFile'] = zf.read()
    else:
        opt['Content']['S3Bucket'] = params['content'].get('s3_bucket')
        opt['Content']['S3Key'] = params['content'].get('s3_key')
        if params['content'].get('s3_object_version') is not None:
            opt['Content']['S3ObjectVersion'] = params['content'].get('s3_object_version')
    try:
        layer_version = lambda_client.publish_layer_version(**opt)
        layer_version.pop('ResponseMetadata', None)
        return {'changed': True, 'layer_versions': [camel_dict_to_snake_dict(layer_version)]}
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise LambdaLayerFailure(e, 'Failed to publish a new layer version (check that you have required permissions).')