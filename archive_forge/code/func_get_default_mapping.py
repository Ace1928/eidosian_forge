from . import errors, registry
from .branch import Branch
from .repository import Repository
from .revision import Revision
def get_default_mapping(self):
    """Get the default mapping for this repository."""
    raise NotImplementedError(self.get_default_mapping)