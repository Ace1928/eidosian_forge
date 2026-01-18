from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
def convert_service_ports(ports):
    if isinstance(ports, list):
        return ports
    if not isinstance(ports, dict):
        raise TypeError('Invalid type for ports, expected dict or list')
    result = []
    for k, v in ports.items():
        port_spec = {'Protocol': 'tcp', 'PublishedPort': k}
        if isinstance(v, tuple):
            port_spec['TargetPort'] = v[0]
            if len(v) >= 2 and v[1] is not None:
                port_spec['Protocol'] = v[1]
            if len(v) == 3:
                port_spec['PublishMode'] = v[2]
            if len(v) > 3:
                raise ValueError('Service port configuration can have at most 3 elements: (target_port, protocol, mode)')
        else:
            port_spec['TargetPort'] = v
        result.append(port_spec)
    return result