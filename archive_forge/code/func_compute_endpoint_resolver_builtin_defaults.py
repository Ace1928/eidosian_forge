import copy
import logging
import socket
import botocore.exceptions
import botocore.parsers
import botocore.serialize
from botocore.config import Config
from botocore.endpoint import EndpointCreator
from botocore.regions import EndpointResolverBuiltins as EPRBuiltins
from botocore.regions import EndpointRulesetResolver
from botocore.signers import RequestSigner
from botocore.useragent import UserAgentString
from botocore.utils import ensure_boolean, is_s3_accelerate_url
def compute_endpoint_resolver_builtin_defaults(self, region_name, service_name, s3_config, endpoint_bridge, client_endpoint_url, legacy_endpoint_url):
    if client_endpoint_url:
        given_endpoint = client_endpoint_url
    elif not endpoint_bridge.resolver_uses_builtin_data():
        given_endpoint = legacy_endpoint_url
    else:
        given_endpoint = None
    if s3_config.get('use_accelerate_endpoint', False):
        force_path_style = False
    elif client_endpoint_url is not None and (not is_s3_accelerate_url(client_endpoint_url)):
        force_path_style = s3_config.get('addressing_style') != 'virtual'
    else:
        force_path_style = s3_config.get('addressing_style') == 'path'
    return {EPRBuiltins.AWS_REGION: region_name, EPRBuiltins.AWS_USE_FIPS: given_endpoint is None and endpoint_bridge._resolve_endpoint_variant_config_var('use_fips_endpoint') or False, EPRBuiltins.AWS_USE_DUALSTACK: given_endpoint is None and endpoint_bridge._resolve_use_dualstack_endpoint(service_name) or False, EPRBuiltins.AWS_STS_USE_GLOBAL_ENDPOINT: self._should_set_global_sts_endpoint(region_name=region_name, endpoint_url=None, endpoint_config=None), EPRBuiltins.AWS_S3_USE_GLOBAL_ENDPOINT: self._should_force_s3_global(region_name, s3_config), EPRBuiltins.AWS_S3_ACCELERATE: s3_config.get('use_accelerate_endpoint', False), EPRBuiltins.AWS_S3_FORCE_PATH_STYLE: force_path_style, EPRBuiltins.AWS_S3_USE_ARN_REGION: s3_config.get('use_arn_region', True), EPRBuiltins.AWS_S3CONTROL_USE_ARN_REGION: s3_config.get('use_arn_region', False), EPRBuiltins.AWS_S3_DISABLE_MRAP: s3_config.get('s3_disable_multiregion_access_points', False), EPRBuiltins.SDK_ENDPOINT: given_endpoint}