from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
def serialize_msg(raw_msg):
    msg = {_VERSION_KEY: _RPC_ENVELOPE_VERSION, _MESSAGE_KEY: jsonutils.dumps(raw_msg)}
    return msg