from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def build_revision(self):
    rev_props = self._legal_revision_properties(self.command.properties)
    if 'branch-nick' not in rev_props:
        rev_props['branch-nick'] = self.cache_mgr.branch_mapper.git_to_bzr(self.branch_ref)
    self._save_author_info(rev_props)
    committer = self.command.committer
    who = self._format_name_email('committer', committer[0], committer[1])
    try:
        message = self.command.message.decode('utf-8')
    except UnicodeDecodeError:
        self.warning('commit message not in utf8 - replacing unknown characters')
        message = self.command.message.decode('utf-8', 'replace')
    if not _serializer_handles_escaping:
        message = helpers.escape_commit_message(message)
    return revision.Revision(timestamp=committer[2], timezone=committer[3], committer=who, message=message, revision_id=self.revision_id, properties=rev_props, parent_ids=self.parents)