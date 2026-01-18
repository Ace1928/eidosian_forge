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
def add_recursion_detection_header(params, **kwargs):
    has_lambda_name = 'AWS_LAMBDA_FUNCTION_NAME' in os.environ
    trace_id = os.environ.get('_X_AMZN_TRACE_ID')
    if has_lambda_name and trace_id:
        headers = params['headers']
        if 'X-Amzn-Trace-Id' not in headers:
            headers['X-Amzn-Trace-Id'] = quote(trace_id, safe='-=;:+&[]{}"\',')