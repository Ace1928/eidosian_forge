import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def retrieve_iam_role_credentials(self):
    try:
        token = self._fetch_metadata_token()
        role_name = self._get_iam_role(token)
        credentials = self._get_credentials(role_name, token)
        if self._contains_all_credential_fields(credentials):
            credentials = {'role_name': role_name, 'access_key': credentials['AccessKeyId'], 'secret_key': credentials['SecretAccessKey'], 'token': credentials['Token'], 'expiry_time': credentials['Expiration']}
            self._evaluate_expiration(credentials)
            return credentials
        else:
            if 'Code' in credentials and 'Message' in credentials:
                logger.debug('Error response received when retrievingcredentials: %s.', credentials)
            return {}
    except self._RETRIES_EXCEEDED_ERROR_CLS:
        logger.debug('Max number of attempts exceeded (%s) when attempting to retrieve data from metadata service.', self._num_attempts)
    except BadIMDSRequestError as e:
        logger.debug('Bad IMDS request: %s', e.request)
    return {}