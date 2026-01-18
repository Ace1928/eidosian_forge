from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def make_remote_dir_robustly(self, relpath, mode=None):
    """Create a remote directory, clearing the way on the remote side.

        When doing a full upload, it may happen that a file exists where we
        want to create our directory.
        """
    try:
        st = self._up_stat(relpath)
        if not stat.S_ISDIR(st.st_mode):
            if not self.quiet:
                self.outf.write('Deleting {}/{}\n'.format(self.to_transport.external_url(), relpath))
            self._up_delete(relpath)
        else:
            return
    except errors.PathError:
        pass
    self.make_remote_dir(relpath, mode)