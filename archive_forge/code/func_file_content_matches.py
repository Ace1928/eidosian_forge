from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def file_content_matches(self, source_path: str, target_path: str, source_stat=None, target_stat=None):
    """Check if two files are the same in the source and target trees.

        This only checks that the contents of the files are the same,
        it does not touch anything else.

        Args:
          source_path: Path of the file in the source tree
          target_path: Path of the file in the target tree
          source_stat: Optional stat value of the file in the source tree
          target_stat: Optional stat value of the file in the target tree
        Returns: Boolean indicating whether the files have the same contents
        """
    with self.lock_read():
        source_verifier_kind, source_verifier_data = self.source.get_file_verifier(source_path, source_stat)
        target_verifier_kind, target_verifier_data = self.target.get_file_verifier(target_path, target_stat)
        if source_verifier_kind == target_verifier_kind:
            return source_verifier_data == target_verifier_data
        if source_verifier_kind != 'SHA1':
            source_sha1 = self.source.get_file_sha1(source_path, source_stat)
        else:
            source_sha1 = source_verifier_data
        if target_verifier_kind != 'SHA1':
            target_sha1 = self.target.get_file_sha1(target_path, target_stat)
        else:
            target_sha1 = target_verifier_data
        return source_sha1 == target_sha1