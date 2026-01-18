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
def remove_bucket_from_url_paths_from_model(params, model, context, **kwargs):
    """Strips leading `{Bucket}/` from any operations that have it.

    The original value is retained in a separate "authPath" field. This is
    used in the HmacV1Auth signer. See HmacV1Auth.canonical_resource in
    botocore/auth.py for details.

    This change is applied to the operation model during the first time the
    operation is invoked and then stays in effect for the lifetime of the
    client object.

    When the ruleset based endpoint resolver is in effect, both the endpoint
    ruleset AND the service model place the bucket name in the final URL.
    The result is an invalid URL. This handler modifies the operation model to
    no longer place the bucket name. Previous versions of botocore fixed the
    URL after the fact when necessary. Since the introduction of ruleset based
    endpoint resolution, the problem exists in ALL URLs that contain a bucket
    name and can therefore be addressed before the URL gets assembled.
    """
    req_uri = model.http['requestUri']
    bucket_path = '/{Bucket}'
    if req_uri.startswith(bucket_path):
        model.http['requestUri'] = req_uri[len(bucket_path):]
        req_uri = req_uri.split('?')[0]
        needs_slash = req_uri == bucket_path
        model.http['authPath'] = f'{req_uri}/' if needs_slash else req_uri