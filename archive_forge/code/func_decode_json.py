import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
def decode_json(raw_data, root_types=(dict,)):
    """Parse raw data to get decoded object.

    Decodes a JSON encoded 'blob' from a given raw data binary string and
    checks that the root type of that decoded object is in the allowed set of
    types (by default a dict should be the root type).
    """
    try:
        data = jsonutils.loads(binary_decode(raw_data))
    except UnicodeDecodeError as e:
        raise ValueError('Expected UTF-8 decodable data: %s' % e)
    except ValueError as e:
        raise ValueError('Expected JSON decodable data: %s' % e)
    else:
        return _check_decoded_type(data, root_types=root_types)