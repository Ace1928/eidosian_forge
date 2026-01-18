import copy
import logging
import re
from enum import Enum
from botocore import UNSIGNED, xform_name
from botocore.auth import AUTH_TYPE_MAPS, HAS_CRT
from botocore.crt import CRT_SUPPORTED_AUTH_TYPES
from botocore.endpoint_provider import EndpointProvider
from botocore.exceptions import (
from botocore.utils import ensure_boolean, instance_cache
def ruleset_error_to_botocore_exception(self, ruleset_exception, params):
    """Attempts to translate ruleset errors to pre-existing botocore
        exception types by string matching exception strings.
        """
    msg = ruleset_exception.kwargs.get('msg')
    if msg is None:
        return
    if msg.startswith('Invalid region in ARN: '):
        try:
            label = msg.split('`')[1]
        except IndexError:
            label = msg
        return InvalidHostLabelError(label=label)
    service_name = self._service_model.service_name
    if service_name == 's3':
        if msg == 'S3 Object Lambda does not support S3 Accelerate' or msg == 'Accelerate cannot be used with FIPS':
            return UnsupportedS3ConfigurationError(msg=msg)
        if msg.startswith('S3 Outposts does not support') or msg.startswith('S3 MRAP does not support') or msg.startswith('S3 Object Lambda does not support') or msg.startswith('Access Points do not support') or msg.startswith('Invalid configuration:') or msg.startswith('Client was configured for partition'):
            return UnsupportedS3AccesspointConfigurationError(msg=msg)
        if msg.lower().startswith('invalid arn:'):
            return ParamValidationError(report=msg)
    if service_name == 's3control':
        if msg.startswith('Invalid ARN:'):
            arn = params.get('Bucket')
            return UnsupportedS3ControlArnError(arn=arn, msg=msg)
        if msg.startswith('Invalid configuration:') or msg.startswith('Client was configured for partition'):
            return UnsupportedS3ControlConfigurationError(msg=msg)
        if msg == 'AccountId is required but not set':
            return ParamValidationError(report=msg)
    if service_name == 'events':
        if msg.startswith('Invalid Configuration: FIPS is not supported with EventBridge multi-region endpoints.'):
            return InvalidEndpointConfigurationError(msg=msg)
        if msg == 'EndpointId must be a valid host label.':
            return InvalidEndpointConfigurationError(msg=msg)
    return None