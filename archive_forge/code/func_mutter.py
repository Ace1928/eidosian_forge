from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def mutter(self, msg, *args):
    """Output a mutter but add context."""
    msg = '{} ({})'.format(msg, self.command.id)
    mutter(msg, *args)