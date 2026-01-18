from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
def get_request_post_data(request_id: RequestId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, str]:
    """
    Returns post data sent with the request. Returns an error when no data was sent with the request.

    :param request_id: Identifier of the network request to get content for.
    :returns: Request body string, omitting files from multipart requests
    """
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Network.getRequestPostData', 'params': params}
    json = (yield cmd_dict)
    return str(json['postData'])