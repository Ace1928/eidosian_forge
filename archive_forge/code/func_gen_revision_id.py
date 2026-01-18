from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def gen_revision_id(self):
    """Generate a revision id.

        Subclasses may override this to produce deterministic ids say.
        """
    committer = self.command.committer
    who = self._format_name_email('committer', committer[0], committer[1])
    timestamp = committer[2]
    return generate_ids.gen_revision_id(who, timestamp)