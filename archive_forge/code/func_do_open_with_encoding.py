import collections
import locale
import sys
from typing import Dict, List
from charset_normalizer import from_path  # pylint: disable=import-error
def do_open_with_encoding(self, filename, mode: str='r'):
    """Return opened file with a specific encoding.

        Parameters
        ----------
        filename : str
            The full path name of the file to open.
        mode : str
            The mode to open the file in.  Defaults to read-only.

        Returns
        -------
        contents : TextIO
            The contents of the file.
        """
    return open(filename, mode=mode, encoding=self.encoding, newline='')