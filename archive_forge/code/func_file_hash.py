import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def file_hash(fname, alg='sha256'):
    """
        Calculate the hash of a given file.
        Useful for checking if a file has changed or been corrupted.
        Parameters
        ----------
        fname : str
            The name of the file.
        alg : str
            The type of the hashing algorithm
        Returns
        -------
        hash : str
            The hash of the file.
        Examples
        --------
        >>> fname = "test-file-for-hash.txt"
        >>> with open(fname, "w") as f:
        ...     __ = f.write("content of the file")
        >>> print(file_hash(fname))
        0fc74468e6a9a829f103d069aeb2bb4f8646bad58bf146bb0e3379b759ec4a00
        >>> import os
        >>> os.remove(fname)
        """
    import hashlib
    if alg not in hashlib.algorithms_available:
        raise ValueError(f"Algorithm '{alg}' not available in hashlib")
    chunksize = 65536
    hasher = hashlib.new(alg)
    with open(fname, 'rb') as fin:
        buff = fin.read(chunksize)
        while buff:
            hasher.update(buff)
            buff = fin.read(chunksize)
    return hasher.hexdigest()