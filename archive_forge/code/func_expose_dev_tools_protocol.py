from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def expose_dev_tools_protocol(target_id: TargetID, binding_name: typing.Optional[str]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Inject object to the target's main frame that provides a communication
    channel with browser target.

    Injected object will be available as ``window[bindingName]``.

    The object has the follwing API:
    - ``binding.send(json)`` - a method to send messages over the remote debugging protocol
    - ``binding.onmessage = json => handleMessage(json)`` - a callback that will be called for the protocol notifications and command responses.

    **EXPERIMENTAL**

    :param target_id:
    :param binding_name: *(Optional)* Binding name, 'cdp' if not specified.
    """
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    if binding_name is not None:
        params['bindingName'] = binding_name
    cmd_dict: T_JSON_DICT = {'method': 'Target.exposeDevToolsProtocol', 'params': params}
    json = (yield cmd_dict)