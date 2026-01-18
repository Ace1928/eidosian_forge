from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
def get_layer_version(lambda_client, layer_name, version_number):
    try:
        layer_version = lambda_client.get_layer_version(LayerName=layer_name, VersionNumber=version_number)
        if layer_version:
            layer_version.pop('ResponseMetadata')
        return [camel_dict_to_snake_dict(layer_version)]
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        raise LambdaLayerInfoFailure(exc=e, msg='get_layer_version() failed.')