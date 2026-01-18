from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def delete_layer_version(lambda_client, params, check_mode=False):
    name = params.get('name')
    version = params.get('version')
    layer_versions = list_layer_versions(lambda_client, name)
    deleted_versions = []
    changed = False
    for layer in layer_versions:
        if version == -1 or layer['version'] == version:
            deleted_versions.append(layer)
            changed = True
            if not check_mode:
                try:
                    lambda_client.delete_layer_version(LayerName=name, VersionNumber=layer['version'])
                except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
                    raise LambdaLayerFailure(e, f'Failed to delete layer version LayerName={name}, VersionNumber={version}.')
    return {'changed': changed, 'layer_versions': deleted_versions}