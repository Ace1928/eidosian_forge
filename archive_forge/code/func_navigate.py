from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def navigate(url: str, referrer: typing.Optional[str]=None, transition_type: typing.Optional[TransitionType]=None, frame_id: typing.Optional[FrameId]=None, referrer_policy: typing.Optional[ReferrerPolicy]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[FrameId, typing.Optional[network.LoaderId], typing.Optional[str]]]:
    """
    Navigates current page to the given URL.

    :param url: URL to navigate the page to.
    :param referrer: *(Optional)* Referrer URL.
    :param transition_type: *(Optional)* Intended transition type.
    :param frame_id: *(Optional)* Frame id to navigate, if not specified navigates the top frame.
    :param referrer_policy: **(EXPERIMENTAL)** *(Optional)* Referrer-policy used for the navigation.
    :returns: A tuple with the following items:

        0. **frameId** - Frame id that has navigated (or failed to navigate)
        1. **loaderId** - *(Optional)* Loader identifier. This is omitted in case of same-document navigation, as the previously committed loaderId would not change.
        2. **errorText** - *(Optional)* User friendly error message, present if and only if navigation has failed.
    """
    params: T_JSON_DICT = dict()
    params['url'] = url
    if referrer is not None:
        params['referrer'] = referrer
    if transition_type is not None:
        params['transitionType'] = transition_type.to_json()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    if referrer_policy is not None:
        params['referrerPolicy'] = referrer_policy.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Page.navigate', 'params': params}
    json = (yield cmd_dict)
    return (FrameId.from_json(json['frameId']), network.LoaderId.from_json(json['loaderId']) if 'loaderId' in json else None, str(json['errorText']) if 'errorText' in json else None)