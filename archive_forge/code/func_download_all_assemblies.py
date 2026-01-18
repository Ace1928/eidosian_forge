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
def download_all_assemblies(self, listfile=None, file_format=None):
    """Retrieve all biological assemblies not in the local PDB copy.

        :type  listfile: str, optional
        :param listfile: file name to which all assembly codes will be written

        :type  file_format: str, optional
        :param file_format: format in which to download the entries. Available
            options are "mmCif" or "pdb". Defaults to mmCif.
        """
    file_format = self._print_default_format_warning(file_format)
    assemblies = self.get_all_assemblies(file_format)
    for pdb_code, assembly_num in assemblies:
        self.retrieve_assembly_file(pdb_code, assembly_num, file_format=file_format)
    if listfile:
        with open(listfile, 'w') as outfile:
            outfile.writelines((f'{pdb_code}.{assembly_num}\n' for x in assemblies))