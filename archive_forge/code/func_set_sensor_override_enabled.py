from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
def set_sensor_override_enabled(enabled: bool, type_: SensorType, metadata: typing.Optional[SensorMetadata]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Overrides a platform sensor of a given type. If ``enabled`` is true, calls to
    Sensor.start() will use a virtual sensor as backend rather than fetching
    data from a real hardware sensor. Otherwise, existing virtual
    sensor-backend Sensor objects will fire an error event and new calls to
    Sensor.start() will attempt to use a real sensor instead.

    **EXPERIMENTAL**

    :param enabled:
    :param type_:
    :param metadata: *(Optional)*
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    params['type'] = type_.to_json()
    if metadata is not None:
        params['metadata'] = metadata.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Emulation.setSensorOverrideEnabled', 'params': params}
    json = (yield cmd_dict)