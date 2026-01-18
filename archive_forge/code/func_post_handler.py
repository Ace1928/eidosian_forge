from .. import (
import stat
def post_handler(self, cmd):
    if not self.keep:
        return
    for blob_id in self.referenced_blobs:
        self._print_command(self.blobs[blob_id])
    self._print_command(self.command)