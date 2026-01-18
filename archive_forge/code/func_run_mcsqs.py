from __future__ import annotations
import os
import tempfile
import warnings
from collections import namedtuple
from pathlib import Path
from shutil import which
from subprocess import Popen, TimeoutExpired
from monty.dev import requires
from pymatgen.core.structure import Structure
@requires(which('mcsqs') and which('str2cif'), 'run_mcsqs requires first installing AT-AT, see https://www.brown.edu/Departments/Engineering/Labs/avdw/atat/')
def run_mcsqs(structure: Structure, clusters: dict[int, float], scaling: int | list[int]=1, search_time: float=60, directory: str | None=None, instances: int | None=None, temperature: float=1, wr: float=1, wn: float=1, wd: float=0.5, tol: float=0.001) -> Sqs:
    """Helper function for calling mcsqs with different arguments
    Args:
        structure (Structure): Disordered pymatgen Structure object
        clusters (dict): Dictionary of cluster interactions with entries in the form
            number of atoms: cutoff in angstroms
        scaling (int or list): Scaling factor to determine supercell. Two options are possible:
                a. (preferred) Scales number of atoms, e.g., for a structure with 8 atoms,
                   scaling=4 would lead to a 32 atom supercell
                b. A sequence of three scaling factors, e.g., [2, 1, 1], which
                   specifies that the supercell should have dimensions 2a x b x c
            Defaults to 1.
        search_time (float): Time spent looking for the ideal SQS in minutes (default: 60)
        directory (str): Directory to run mcsqs calculation and store files (default: None
            runs calculations in a temp directory)
        instances (int): Specifies the number of parallel instances of mcsqs to run
            (default: number of cpu cores detected by Python)
        temperature (float): Monte Carlo temperature (default: 1), "T" in atat code
        wr (float): Weight assigned to range of perfect correlation match in objective
            function (default = 1)
        wn (float): Multiplicative decrease in weight per additional point in cluster (default: 1)
        wd (float): Exponent of decay in weight as function of cluster diameter (default: 0.5)
        tol (float): Tolerance for matching correlations (default: 1e-3).

    Returns:
        tuple: Pymatgen structure SQS of the input structure, the mcsqs objective function,
            list of all SQS structures, and the directory where calculations are run
    """
    n_atoms = len(structure)
    if structure.is_ordered:
        raise ValueError('Pick a disordered structure')
    if instances is None:
        instances = os.cpu_count()
    original_directory = os.getcwd()
    directory = directory or tempfile.mkdtemp()
    os.chdir(directory)
    if isinstance(scaling, (int, float)):
        if scaling % 1 != 0:
            raise ValueError(f'scaling={scaling!r} should be an integer')
        mcsqs_find_sqs_cmd = ['mcsqs', f'-n {scaling * n_atoms}']
    else:
        with open('sqscell.out', mode='w') as file:
            file.write('1\n1 0 0\n0 1 0\n0 0 1\n')
        structure = structure * scaling
        mcsqs_find_sqs_cmd = ['mcsqs', '-rc', f'-n {n_atoms}']
    structure.to(filename='rndstr.in')
    mcsqs_generate_clusters_cmd = ['mcsqs']
    for num in clusters:
        mcsqs_generate_clusters_cmd.append(f'-{num}={clusters[num]}')
    with Popen(mcsqs_generate_clusters_cmd) as process:
        process.communicate()
    add_ons = [f'-T {temperature}', f'-wr {wr}', f'-wn {wn}', f'-wd {wd}', f'-tol {tol}']
    mcsqs_find_sqs_processes = []
    if instances and instances > 1:
        for i in range(instances):
            instance_cmd = [f'-ip {i + 1}']
            cmd = mcsqs_find_sqs_cmd + add_ons + instance_cmd
            process = Popen(cmd)
            mcsqs_find_sqs_processes.append(process)
    else:
        cmd = mcsqs_find_sqs_cmd + add_ons
        process = Popen(cmd)
        mcsqs_find_sqs_processes.append(process)
    try:
        for process in mcsqs_find_sqs_processes:
            process.communicate(timeout=search_time * 60)
        if instances and instances > 1:
            process = Popen(['mcsqs', '-best'])
            process.communicate()
        if os.path.isfile('bestsqs.out') and os.path.isfile('bestcorr.out'):
            return _parse_sqs_path('.')
        raise RuntimeError('mcsqs exited before timeout reached')
    except TimeoutExpired:
        for process in mcsqs_find_sqs_processes:
            process.kill()
            process.communicate()
        if instances and instances > 1:
            if not os.path.isfile('bestcorr1.out'):
                raise RuntimeError('mcsqs did not generate output files, is search_time sufficient or are number of instances too high?')
            process = Popen(['mcsqs', '-best'])
            process.communicate()
        if os.path.isfile('bestsqs.out') and os.path.isfile('bestcorr.out'):
            return _parse_sqs_path('.')
        os.chdir(original_directory)
        raise TimeoutError('Cluster expansion took too long.')