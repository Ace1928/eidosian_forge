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
def compute_s3_config(self, client_config):
    s3_configuration = self._config_store.get_config_variable('s3')
    if client_config is not None:
        if client_config.s3 is not None:
            if s3_configuration is None:
                s3_configuration = client_config.s3
            else:
                s3_configuration = s3_configuration.copy()
                s3_configuration.update(client_config.s3)
    return s3_configuration