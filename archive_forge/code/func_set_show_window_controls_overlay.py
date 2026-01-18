from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import page
from . import runtime
def set_show_window_controls_overlay(window_controls_overlay_config: typing.Optional[WindowControlsOverlayConfig]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Show Window Controls Overlay for PWA

    :param window_controls_overlay_config: *(Optional)* Window Controls Overlay data, null means hide Window Controls Overlay
    """
    params: T_JSON_DICT = dict()
    if window_controls_overlay_config is not None:
        params['windowControlsOverlayConfig'] = window_controls_overlay_config.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Overlay.setShowWindowControlsOverlay', 'params': params}
    json = (yield cmd_dict)