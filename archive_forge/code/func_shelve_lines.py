import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
def shelve_lines(self, file_id, new_lines):
    """Shelve text changes to a file, using provided lines.

        :param file_id: The file id of the file to shelve the text of.
        :param new_lines: The lines that the file should have due to shelving.
        """
    w_trans_id = self.work_transform.trans_id_file_id(file_id)
    self.work_transform.delete_contents(w_trans_id)
    self.work_transform.create_file(new_lines, w_trans_id)
    s_trans_id = self.shelf_transform.trans_id_file_id(file_id)
    self.shelf_transform.delete_contents(s_trans_id)
    inverse_lines = self._inverse_lines(new_lines, file_id)
    self.shelf_transform.create_file(inverse_lines, s_trans_id)