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
def retrieve_pdb_file(self, pdb_code, obsolete=False, pdir=None, file_format=None, overwrite=False):
    """Fetch PDB structure file from PDB server, and store it locally.

        The PDB structure's file name is returned as a single string.
        If obsolete ``==`` True, the file will be saved in a special file tree.

        NOTE. The default download format has changed from PDB to PDBx/mmCif

        :param pdb_code: 4-symbols structure Id from PDB (e.g. 3J92).
        :type pdb_code: string

        :param file_format:
            File format. Available options:

            * "mmCif" (default, PDBx/mmCif file),
            * "pdb" (format PDB),
            * "xml" (PDBML/XML format),
            * "mmtf" (highly compressed),
            * "bundle" (PDB formatted archive for large structure)

        :type file_format: string

        :param overwrite: if set to True, existing structure files will be overwritten. Default: False
        :type overwrite: bool

        :param obsolete:
            Has a meaning only for obsolete structures. If True, download the obsolete structure
            to 'obsolete' folder, otherwise download won't be performed.
            This option doesn't work for mmtf format as obsoleted structures aren't stored in mmtf.
            Also doesn't have meaning when parameter pdir is specified.
            Note: make sure that you are about to download the really obsolete structure.
            Trying to download non-obsolete structure into obsolete folder will not work
            and you face the "structure doesn't exists" error.
            Default: False

        :type obsolete: bool

        :param pdir: put the file in this directory (default: create a PDB-style directory tree)
        :type pdir: string

        :return: filename
        :rtype: string
        """
    file_format = self._print_default_format_warning(file_format)
    pdb_code = pdb_code.lower()
    archive = {'pdb': f'pdb{pdb_code}.ent.gz', 'mmCif': f'{pdb_code}.cif.gz', 'xml': f'{pdb_code}.xml.gz', 'mmtf': f'{pdb_code}', 'bundle': f'{pdb_code}-pdb-bundle.tar.gz'}
    archive_fn = archive[file_format]
    if file_format not in archive.keys():
        raise Exception(f"Specified file_format {file_format} doesn't exists or is not supported. Maybe a typo. Please, use one of the following: mmCif, pdb, xml, mmtf, bundle")
    if file_format in ('pdb', 'mmCif', 'xml'):
        pdb_dir = 'divided' if not obsolete else 'obsolete'
        file_type = 'pdb' if file_format == 'pdb' else 'mmCIF' if file_format == 'mmCif' else 'XML'
        url = self.pdb_server + f'/pub/pdb/data/structures/{pdb_dir}/{file_type}/{pdb_code[1:3]}/{archive_fn}'
    elif file_format == 'bundle':
        url = self.pdb_server + f'/pub/pdb/compatible/pdb_bundle/{pdb_code[1:3]}/{pdb_code}/{archive_fn}'
    else:
        url = f'http://mmtf.rcsb.org/v1.0/full/{pdb_code}'
    if pdir is None:
        path = self.local_pdb if not obsolete else self.obsolete_pdb
        if not self.flat_tree:
            path = os.path.join(path, pdb_code[1:3])
    else:
        path = pdir
    if not os.access(path, os.F_OK):
        os.makedirs(path)
    filename = os.path.join(path, archive_fn)
    final = {'pdb': f'pdb{pdb_code}.ent', 'mmCif': f'{pdb_code}.cif', 'xml': f'{pdb_code}.xml', 'mmtf': f'{pdb_code}.mmtf', 'bundle': f'{pdb_code}-pdb-bundle.tar'}
    final_file = os.path.join(path, final[file_format])
    if not overwrite:
        if os.path.exists(final_file):
            if self._verbose:
                print(f"Structure exists: '{final_file}' ")
            return final_file
    if self._verbose:
        print(f"Downloading PDB structure '{pdb_code}'...")
    try:
        urlcleanup()
        urlretrieve(url, filename)
    except OSError:
        print("Desired structure doesn't exist")
    else:
        with gzip.open(filename, 'rb') as gz:
            with open(final_file, 'wb') as out:
                out.writelines(gz)
        os.remove(filename)
    return final_file