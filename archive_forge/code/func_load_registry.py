import os
import time
import contextlib
from pathlib import Path
import shlex
import shutil
from .hashes import hash_matches, file_hash
from .utils import (
from .downloaders import DOIDownloader, choose_downloader, doi_to_repository
def load_registry(self, fname):
    """
        Load entries from a file and add them to the registry.

        Use this if you are managing many files.

        Each line of the file should have file name and its hash separated by
        a space. Hash can specify checksum algorithm using "alg:hash" format.
        In case no algorithm is provided, SHA256 is used by default.
        Only one file per line is allowed. Custom download URLs for individual
        files can be specified as a third element on the line. Line comments
        can be added and must be prepended with ``#``.

        Parameters
        ----------
        fname : str | fileobj
            Path (or open file object) to the registry file.

        """
    with contextlib.ExitStack() as stack:
        if hasattr(fname, 'read'):
            fin = fname
        else:
            fin = stack.enter_context(open(fname, encoding='utf-8'))
        for linenum, line in enumerate(fin):
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            line = line.strip()
            if line.startswith('#'):
                continue
            elements = shlex.split(line)
            if not len(elements) in [0, 2, 3]:
                raise OSError(f"Invalid entry in Pooch registry file '{fname}': expected 2 or 3 elements in line {linenum + 1} but got {len(elements)}. Offending entry: '{line}'")
            if elements:
                file_name = elements[0]
                file_checksum = elements[1]
                if len(elements) == 3:
                    file_url = elements[2]
                    self.urls[file_name] = file_url
                self.registry[file_name] = file_checksum.lower()