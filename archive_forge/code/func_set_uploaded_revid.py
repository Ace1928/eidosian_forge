from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def set_uploaded_revid(self, rev_id):
    revid_path = self.branch.get_config_stack().get('upload_revid_location')
    self.to_transport.put_bytes(urlutils.escape(revid_path), rev_id)
    self._uploaded_revid = rev_id