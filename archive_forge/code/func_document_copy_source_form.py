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
def document_copy_source_form(section, event_name, **kwargs):
    if 'request-example' in event_name:
        parent = section.get_section('structure-value')
        param_line = parent.get_section('CopySource')
        value_portion = param_line.get_section('member-value')
        value_portion.clear_text()
        value_portion.write("'string' or {'Bucket': 'string', 'Key': 'string', 'VersionId': 'string'}")
    elif 'request-params' in event_name:
        param_section = section.get_section('CopySource')
        type_section = param_section.get_section('param-type')
        type_section.clear_text()
        type_section.write(':type CopySource: str or dict')
        doc_section = param_section.get_section('param-documentation')
        doc_section.clear_text()
        doc_section.write("The name of the source bucket, key name of the source object, and optional version ID of the source object.  You can either provide this value as a string or a dictionary.  The string form is {bucket}/{key} or {bucket}/{key}?versionId={versionId} if you want to copy a specific version.  You can also provide this value as a dictionary.  The dictionary format is recommended over the string format because it is more explicit.  The dictionary format is: {'Bucket': 'bucket', 'Key': 'key', 'VersionId': 'id'}.  Note that the VersionId key is optional and may be omitted. To specify an S3 access point, provide the access point ARN for the ``Bucket`` key in the copy source dictionary. If you want to provide the copy source for an S3 access point as a string instead of a dictionary, the ARN provided must be the full S3 access point object ARN (i.e. {accesspoint_arn}/object/{key})")