import os
import tempfile
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
def run_naccess(model, pdb_file, probe_size=None, z_slice=None, naccess='naccess', temp_path='/tmp/'):
    """Run naccess for a pdb file."""
    tmp_path = tempfile.mkdtemp(dir=temp_path)
    handle, tmp_pdb_file = tempfile.mkstemp('.pdb', dir=tmp_path)
    os.close(handle)
    if pdb_file:
        pdb_file = os.path.abspath(pdb_file)
        shutil.copy(pdb_file, tmp_pdb_file)
    else:
        writer = PDBIO()
        writer.set_structure(model.get_parent())
        writer.save(tmp_pdb_file)
    old_dir = os.getcwd()
    os.chdir(tmp_path)
    command = [naccess, tmp_pdb_file]
    if probe_size:
        command.extend(['-p', probe_size])
    if z_slice:
        command.extend(['-z', z_slice])
    p = subprocess.Popen(command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    os.chdir(old_dir)
    rsa_file = tmp_pdb_file[:-4] + '.rsa'
    asa_file = tmp_pdb_file[:-4] + '.asa'
    if err.strip():
        warnings.warn(err)
    if not os.path.exists(rsa_file) or not os.path.exists(asa_file):
        raise Exception('NACCESS did not execute or finish properly.')
    with open(rsa_file) as rf:
        rsa_data = rf.readlines()
    with open(asa_file) as af:
        asa_data = af.readlines()
    return (rsa_data, asa_data)