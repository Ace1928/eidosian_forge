import os
import shutil
import stat
import tarfile
import zipfile
from django.core.exceptions import SuspiciousOperation
def has_leading_dir(self, paths):
    """
        Return True if all the paths have the same leading path name
        (i.e., everything is in one subdirectory in an archive).
        """
    common_prefix = None
    for path in paths:
        prefix, rest = self.split_leading_dir(path)
        if not prefix:
            return False
        elif common_prefix is None:
            common_prefix = prefix
        elif prefix != common_prefix:
            return False
    return True