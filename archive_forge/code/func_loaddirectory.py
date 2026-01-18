from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def loaddirectory(self, sect):
    """
        Load the directory.

        :param sect: sector index of directory stream.
        """
    log.debug('Loading the Directory:')
    self.directory_fp = self._open(sect, force_FAT=True)
    max_entries = self.directory_fp.size // 128
    log.debug('loaddirectory: size=%d, max_entries=%d' % (self.directory_fp.size, max_entries))
    self.direntries = [None] * max_entries
    root_entry = self._load_direntry(0)
    self.root = self.direntries[0]
    self.root.build_storage_tree()