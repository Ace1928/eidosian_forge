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
def document_glacier_tree_hash_checksum():
    doc = "\n        This is a required field.\n\n        Ideally you will want to compute this value with checksums from\n        previous uploaded parts, using the algorithm described in\n        `Glacier documentation <http://docs.aws.amazon.com/amazonglacier/latest/dev/checksum-calculations.html>`_.\n\n        But if you prefer, you can also use botocore.utils.calculate_tree_hash()\n        to compute it from raw file by::\n\n            checksum = calculate_tree_hash(open('your_file.txt', 'rb'))\n\n        "
    return AppendParamDocumentation('checksum', doc).append_documentation