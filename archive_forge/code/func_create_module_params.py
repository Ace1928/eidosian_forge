from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def create_module_params():
    """
    Reads the module parameters and returns a dict
    :return: dict
    """
    endpoint_parameters = dict(EndpointIdentifier=module.params.get('endpointidentifier'), EndpointType=module.params.get('endpointtype'), EngineName=module.params.get('enginename'), Username=module.params.get('username'), Password=module.params.get('password'), ServerName=module.params.get('servername'), Port=module.params.get('port'), DatabaseName=module.params.get('databasename'), SslMode=module.params.get('sslmode'))
    if module.params.get('EndpointArn'):
        endpoint_parameters['EndpointArn'] = module.params.get('EndpointArn')
    if module.params.get('certificatearn'):
        endpoint_parameters['CertificateArn'] = module.params.get('certificatearn')
    if module.params.get('dmstransfersettings'):
        endpoint_parameters['DmsTransferSettings'] = module.params.get('dmstransfersettings')
    if module.params.get('extraconnectionattributes'):
        endpoint_parameters['ExtraConnectionAttributes'] = module.params.get('extraconnectionattributes')
    if module.params.get('kmskeyid'):
        endpoint_parameters['KmsKeyId'] = module.params.get('kmskeyid')
    if module.params.get('tags'):
        endpoint_parameters['Tags'] = module.params.get('tags')
    if module.params.get('serviceaccessrolearn'):
        endpoint_parameters['ServiceAccessRoleArn'] = module.params.get('serviceaccessrolearn')
    if module.params.get('externaltabledefinition'):
        endpoint_parameters['ExternalTableDefinition'] = module.params.get('externaltabledefinition')
    if module.params.get('dynamodbsettings'):
        endpoint_parameters['DynamoDbSettings'] = module.params.get('dynamodbsettings')
    if module.params.get('s3settings'):
        endpoint_parameters['S3Settings'] = module.params.get('s3settings')
    if module.params.get('mongodbsettings'):
        endpoint_parameters['MongoDbSettings'] = module.params.get('mongodbsettings')
    if module.params.get('kinesissettings'):
        endpoint_parameters['KinesisSettings'] = module.params.get('kinesissettings')
    if module.params.get('elasticsearchsettings'):
        endpoint_parameters['ElasticsearchSettings'] = module.params.get('elasticsearchsettings')
    if module.params.get('wait'):
        endpoint_parameters['wait'] = module.boolean(module.params.get('wait'))
    if module.params.get('timeout'):
        endpoint_parameters['timeout'] = module.params.get('timeout')
    if module.params.get('retries'):
        endpoint_parameters['retries'] = module.params.get('retries')
    return endpoint_parameters