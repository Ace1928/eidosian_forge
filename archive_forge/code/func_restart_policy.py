from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@property
def restart_policy(self):
    return self.get('RestartPolicy')