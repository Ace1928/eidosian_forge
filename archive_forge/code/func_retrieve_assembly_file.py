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
def retrieve_assembly_file(self, pdb_code, assembly_num, pdir=None, file_format=None, overwrite=False):
    """Fetch one or more assembly structures associated with a PDB entry.

        Unless noted below, parameters are described in ``retrieve_pdb_file``.

        :type  assembly_num: int
        :param assembly_num: assembly number to download.

        :rtype : str
        :return: file name of the downloaded assembly file.
        """
    pdb_code = pdb_code.lower()
    assembly_num = int(assembly_num)
    archive = {'pdb': f'{pdb_code}.pdb{assembly_num}.gz', 'mmcif': f'{pdb_code}-assembly{assembly_num}.cif.gz'}
    file_format = self._print_default_format_warning(file_format)
    file_format = file_format.lower()
    if file_format not in archive:
        raise Exception(f"Specified file_format '{file_format}' is not supported. Use one of the following: 'mmcif' or 'pdb'.")
    archive_fn = archive[file_format]
    if file_format == 'mmcif':
        url = self.pdb_server + f'/pub/pdb/data/assemblies/mmCIF/all/{archive_fn}'
    elif file_format == 'pdb':
        url = self.pdb_server + f'/pub/pdb/data/biounit/PDB/all/{archive_fn}'
    else:
        raise ValueError(f"file_format '{file_format}' not supported")
    if pdir is None:
        path = self.local_pdb
        if not self.flat_tree:
            path = os.path.join(path, pdb_code[1:3])
    else:
        path = pdir
    if not os.access(path, os.F_OK):
        os.makedirs(path)
    assembly_gz_file = os.path.join(path, archive_fn)
    assembly_final_file = os.path.join(path, archive_fn[:-3])
    if not overwrite:
        if os.path.exists(assembly_final_file):
            if self._verbose:
                print(f"Structure exists: '{assembly_final_file}' ")
            return assembly_final_file
    if self._verbose:
        print(f"Downloading assembly ({assembly_num}) for PDB entry '{pdb_code}'...")
    try:
        urlcleanup()
        urlretrieve(url, assembly_gz_file)
    except OSError as err:
        print(f'Download failed! Maybe the desired assembly does not exist: {err}')
    else:
        with gzip.open(assembly_gz_file, 'rb') as gz:
            with open(assembly_final_file, 'wb') as out:
                out.writelines(gz)
        os.remove(assembly_gz_file)
    return assembly_final_file