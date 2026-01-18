from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def build_storage_tree(self):
    """
        Read and build the red-black tree attached to this OleDirectoryEntry
        object, if it is a storage.
        Note that this method builds a tree of all subentries, so it should
        only be called for the root object once.
        """
    log.debug('build_storage_tree: SID=%d - %s - sid_child=%d' % (self.sid, repr(self.name), self.sid_child))
    if self.sid_child != NOSTREAM:
        self.append_kids(self.sid_child)
        self.kids.sort()