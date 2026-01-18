import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_change(self, change):
    """Shelve a change in the iter_shelvable format."""
    if change[0] == 'rename':
        self.shelve_rename(change[1])
    elif change[0] == 'delete file':
        self.shelve_deletion(change[1])
    elif change[0] == 'add file':
        self.shelve_creation(change[1])
    elif change[0] in ('change kind', 'modify text'):
        self.shelve_content_change(change[1])
    elif change[0] == 'modify target':
        self.shelve_modify_target(change[1])
    else:
        raise ValueError('Unknown change kind: "%s"' % change[0])