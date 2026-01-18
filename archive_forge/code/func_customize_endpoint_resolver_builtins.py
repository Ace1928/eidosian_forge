import base64
import copy
import logging
import os
import re
import uuid
import warnings
from io import BytesIO
import botocore
import botocore.auth
from botocore import utils
from botocore.compat import (
from botocore.docs.utils import (
from botocore.endpoint_provider import VALID_HOST_LABEL_RE
from botocore.exceptions import (
from botocore.regions import EndpointResolverBuiltins
from botocore.signers import (
from botocore.utils import (
from botocore import retryhandler  # noqa
from botocore import translate  # noqa
from botocore.compat import MD5_AVAILABLE  # noqa
from botocore.exceptions import MissingServiceIdError  # noqa
from botocore.utils import hyphenize_service_id  # noqa
from botocore.utils import is_global_accesspoint  # noqa
from botocore.utils import SERVICE_NAME_ALIASES  # noqa
def customize_endpoint_resolver_builtins(builtins, model, params, context, **kwargs):
    """Modify builtin parameter values for endpoint resolver

    Modifies the builtins dict in place. Changes are in effect for one call.
    The corresponding event is emitted only if at least one builtin parameter
    value is required for endpoint resolution for the operation.
    """
    bucket_name = params.get('Bucket')
    bucket_is_arn = bucket_name is not None and ArnParser.is_arn(bucket_name)
    if model.name == 'GetBucketLocation':
        builtins[EndpointResolverBuiltins.AWS_S3_FORCE_PATH_STYLE] = True
    elif bucket_is_arn:
        builtins[EndpointResolverBuiltins.AWS_S3_FORCE_PATH_STYLE] = False
    path_style_required = bucket_name is not None and (not VALID_HOST_LABEL_RE.match(bucket_name))
    path_style_requested = builtins[EndpointResolverBuiltins.AWS_S3_FORCE_PATH_STYLE]
    if context.get('use_global_endpoint') and (not path_style_required) and (not path_style_requested) and (not bucket_is_arn) and (not utils.is_s3express_bucket(bucket_name)):
        builtins[EndpointResolverBuiltins.AWS_REGION] = 'aws-global'
        builtins[EndpointResolverBuiltins.AWS_S3_USE_GLOBAL_ENDPOINT] = True