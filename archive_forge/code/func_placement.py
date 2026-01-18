from .. import errors
from ..constants import IS_WINDOWS_PLATFORM
from ..utils import (
@property
def placement(self):
    return self.get('Placement')