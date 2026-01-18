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
def handle_copy_source_param(params, **kwargs):
    """Convert CopySource param for CopyObject/UploadPartCopy.

    This handler will deal with two cases:

        * CopySource provided as a string.  We'll make a best effort
          to URL encode the key name as required.  This will require
          parsing the bucket and version id from the CopySource value
          and only encoding the key.
        * CopySource provided as a dict.  In this case we're
          explicitly given the Bucket, Key, and VersionId so we're
          able to encode the key and ensure this value is serialized
          and correctly sent to S3.

    """
    source = params.get('CopySource')
    if source is None:
        return
    if isinstance(source, str):
        params['CopySource'] = _quote_source_header(source)
    elif isinstance(source, dict):
        params['CopySource'] = _quote_source_header_from_dict(source)