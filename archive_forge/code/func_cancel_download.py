from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def cancel_download(guid: str, browser_context_id: typing.Optional[BrowserContextID]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Cancel a download if in progress

    **EXPERIMENTAL**

    :param guid: Global unique identifier of the download.
    :param browser_context_id: *(Optional)* BrowserContext to perform the action in. When omitted, default browser context is used.
    """
    params: T_JSON_DICT = dict()
    params['guid'] = guid
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Browser.cancelDownload', 'params': params}
    json = (yield cmd_dict)