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
def add_retry_headers(request, **kwargs):
    retries_context = request.context.get('retries')
    if not retries_context:
        return
    headers = request.headers
    headers['amz-sdk-invocation-id'] = retries_context['invocation-id']
    sdk_retry_keys = ('ttl', 'attempt', 'max')
    sdk_request_headers = [f'{key}={retries_context[key]}' for key in sdk_retry_keys if key in retries_context]
    headers['amz-sdk-request'] = '; '.join(sdk_request_headers)