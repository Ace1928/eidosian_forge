from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@classmethod
def parse_mount_string(cls, string):
    parts = string.split(':')
    if len(parts) > 3:
        raise errors.InvalidArgument(f'Invalid mount format "{string}"')
    if len(parts) == 1:
        return cls(target=parts[0], source=None)
    else:
        target = parts[1]
        source = parts[0]
        mount_type = 'volume'
        if source.startswith('/') or (IS_WINDOWS_PLATFORM and source[0].isalpha() and (source[1] == ':')):
            mount_type = 'bind'
        read_only = not (len(parts) == 2 or parts[2] == 'rw')
        return cls(target, source, read_only=read_only, type=mount_type)