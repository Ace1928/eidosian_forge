from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def get_revision_id_to_revno_map(self):
    """Return the revision_id => dotted revno map.

        This will be regenerated on demand, but will be cached.

        Returns: A dictionary mapping revision_id => dotted revno.
            This dictionary should not be modified by the caller.
        """
    if 'evil' in debug.debug_flags:
        mutter_callsite(3, 'get_revision_id_to_revno_map scales with ancestry.')
    with self.lock_read():
        if self._revision_id_to_revno_cache is not None:
            mapping = self._revision_id_to_revno_cache
        else:
            mapping = self._gen_revno_map()
            self._cache_revision_id_to_revno(mapping)
        return mapping