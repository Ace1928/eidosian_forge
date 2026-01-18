from collections.abc import Mapping
import copy
import logging
import sys
import traceback
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import oslo_messaging
from oslo_messaging import _utils as utils
def serialize_remote_exception(failure_info):
    """Prepares exception data to be sent over rpc.

    Failure_info should be a sys.exc_info() tuple.

    """
    tb = traceback.format_exception(*failure_info)
    failure = failure_info[1]
    kwargs = {}
    if hasattr(failure, 'kwargs'):
        kwargs = failure.kwargs
    cls_name = str(failure.__class__.__name__)
    mod_name = str(failure.__class__.__module__)
    if cls_name.endswith(_REMOTE_POSTFIX) and mod_name.endswith(_REMOTE_POSTFIX):
        cls_name = cls_name[:-len(_REMOTE_POSTFIX)]
        mod_name = mod_name[:-len(_REMOTE_POSTFIX)]
    data = {'class': cls_name, 'module': mod_name, 'message': str(failure), 'tb': tb, 'args': failure.args, 'kwargs': kwargs}
    json_data = jsonutils.dumps(data)
    return json_data