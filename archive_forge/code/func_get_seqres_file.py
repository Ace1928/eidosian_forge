import contextlib
import ftplib
import gzip
import os
import re
import shutil
import sys
from urllib.request import urlopen
from urllib.request import urlretrieve
from urllib.request import urlcleanup
def get_seqres_file(self, savefile='pdb_seqres.txt'):
    """Retrieve and save a (big) file containing all the sequences of PDB entries."""
    if self._verbose:
        print('Retrieving sequence file (takes over 110 MB).')
    url = self.pdb_server + '/pub/pdb/derived_data/pdb_seqres.txt'
    urlretrieve(url, savefile)