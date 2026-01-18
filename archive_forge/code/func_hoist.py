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
def hoist(self, params, **kwargs):
    """Hoist a header to the hostname.

        Hoist a header to the beginning of the hostname with a suffix "." after
        it. The original header should be removed from the header map. This
        method is intended to be used as a target for the before-call event.
        """
    if self._header_name not in params['headers']:
        return
    header_value = params['headers'][self._header_name]
    self._ensure_header_is_valid_host(header_value)
    original_url = params['url']
    new_url = self._prepend_to_host(original_url, header_value)
    params['url'] = new_url