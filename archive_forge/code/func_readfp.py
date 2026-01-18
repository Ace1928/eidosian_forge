import os
import sys
import posixpath
import urllib.parse
def readfp(self, fp, strict=True):
    """
        Read a single mime.types-format file.

        If strict is true, information will be added to
        list of standard types, else to the list of non-standard
        types.
        """
    while 1:
        line = fp.readline()
        if not line:
            break
        words = line.split()
        for i in range(len(words)):
            if words[i][0] == '#':
                del words[i:]
                break
        if not words:
            continue
        type, suffixes = (words[0], words[1:])
        for suff in suffixes:
            self.add_type(type, '.' + suff, strict)