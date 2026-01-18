import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def get_disk_name(self, branch):
    """Generate a suitable basename for storing this directive on disk

        :param branch: The Branch this merge directive was generated fro
        :return: A string
        """
    revno, revision_id = branch.last_revision_info()
    if self.revision_id == revision_id:
        revno = [revno]
    else:
        try:
            revno = branch.revision_id_to_dotted_revno(self.revision_id)
        except errors.NoSuchRevision:
            revno = ['merge']
    nick = re.sub('(\\W+)', '-', branch.nick).strip('-')
    return '{}-{}'.format(nick, '.'.join((str(n) for n in revno)))