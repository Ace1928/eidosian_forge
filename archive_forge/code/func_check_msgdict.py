from numbers import Integral, Real
from .specs import (
def check_msgdict(msgdict):
    spec = SPEC_BY_TYPE.get(msgdict['type'])
    if spec is None:
        raise ValueError('unknown message type {!r}'.format(msgdict['type']))
    for name, value in msgdict.items():
        if name not in spec['attribute_names']:
            raise ValueError('{} message has no attribute {}'.format(spec['type'], name))
        check_value(name, value)