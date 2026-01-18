import os
from inspect import isclass
from copy import deepcopy
from warnings import warn
from packaging.version import Version
from traits.trait_errors import TraitError
from traits.trait_handlers import TraitDictObject, TraitListObject
from ...utils.filemanip import md5, hash_infile, hash_timestamp
from .traits_extension import (
from ... import config, __version__
def get_hashval(self, hash_method=None):
    """Return a dictionary of our items with hashes for each file.

        Searches through dictionary items and if an item is a file, it
        calculates the md5 hash of the file contents and stores the
        file name and hash value as the new key value.

        However, the overall bunch hash is calculated only on the hash
        value of a file. The path and name of the file are not used in
        the overall hash calculation.

        Returns
        -------
        list_withhash : dict
            Copy of our dictionary with the new file hashes included
            with each file.
        hashvalue : str
            The md5 hash value of the traited spec

        """
    list_withhash = []
    list_nofilename = []
    for name, val in sorted(self.trait_get().items()):
        if not isdefined(val) or self.has_metadata(name, 'nohash', True):
            continue
        hash_files = not self.has_metadata(name, 'hash_files', False) and (not self.has_metadata(name, 'name_source'))
        list_nofilename.append((name, self._get_sorteddict(val, hash_method=hash_method, hash_files=hash_files)))
        list_withhash.append((name, self._get_sorteddict(val, True, hash_method=hash_method, hash_files=hash_files)))
    return (list_withhash, md5(str(list_nofilename).encode()).hexdigest())