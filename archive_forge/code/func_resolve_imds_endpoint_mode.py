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
def resolve_imds_endpoint_mode(session):
    """Resolving IMDS endpoint mode to either IPv6 or IPv4.

    ec2_metadata_service_endpoint_mode takes precedence over imds_use_ipv6.
    """
    endpoint_mode = session.get_config_variable('ec2_metadata_service_endpoint_mode')
    if endpoint_mode is not None:
        lendpoint_mode = endpoint_mode.lower()
        if lendpoint_mode not in METADATA_ENDPOINT_MODES:
            error_msg_kwargs = {'mode': endpoint_mode, 'valid_modes': METADATA_ENDPOINT_MODES}
            raise InvalidIMDSEndpointModeError(**error_msg_kwargs)
        return lendpoint_mode
    elif session.get_config_variable('imds_use_ipv6'):
        return 'ipv6'
    return 'ipv4'