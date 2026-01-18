from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@property
def replicas(self):
    if 'replicated' in self:
        return self['replicated'].get('Replicas')
    if 'ReplicatedJob' in self:
        return self['ReplicatedJob'].get('TotalCompletions')
    return None