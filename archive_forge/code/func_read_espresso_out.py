import os
import operator as op
import re
import warnings
from collections import OrderedDict
from os import path
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from ase.calculators.calculator import kpts2ndarray, kpts2sizeandoffsets
from ase.dft.kpoints import kpoint_convert
from ase.constraints import FixAtoms, FixCartesian
from ase.data import chemical_symbols, atomic_numbers
from ase.units import create_units
from ase.utils import iofunction
@iofunction('rU')
def read_espresso_out(fileobj, index=-1, results_required=True):
    """Reads Quantum ESPRESSO output files.

    The atomistic configurations as well as results (energy, force, stress,
    magnetic moments) of the calculation are read for all configurations
    within the output file.

    Will probably raise errors for broken or incomplete files.

    Parameters
    ----------
    fileobj : file|str
        A file like object or filename
    index : slice
        The index of configurations to extract.
    results_required : bool
        If True, atomistic configurations that do not have any
        associated results will not be included. This prevents double
        printed configurations and incomplete calculations from being
        returned as the final configuration with no results data.

    Yields
    ------
    structure : Atoms
        The next structure from the index slice. The Atoms has a
        SinglePointCalculator attached with any results parsed from
        the file.


    """
    pwo_lines = fileobj.readlines()
    indexes = {_PW_START: [], _PW_END: [], _PW_CELL: [], _PW_POS: [], _PW_MAGMOM: [], _PW_FORCE: [], _PW_TOTEN: [], _PW_STRESS: [], _PW_FERMI: [], _PW_HIGHEST_OCCUPIED: [], _PW_HIGHEST_OCCUPIED_LOWEST_FREE: [], _PW_KPTS: [], _PW_BANDS: [], _PW_BANDSTRUCTURE: []}
    for idx, line in enumerate(pwo_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)
    all_config_indexes = sorted(indexes[_PW_START] + indexes[_PW_POS])
    if results_required:
        results_indexes = sorted(indexes[_PW_TOTEN] + indexes[_PW_FORCE] + indexes[_PW_STRESS] + indexes[_PW_MAGMOM] + indexes[_PW_BANDS] + indexes[_PW_BANDSTRUCTURE])
        results_config_indexes = []
        for config_index, config_index_next in zip(all_config_indexes, all_config_indexes[1:] + [len(pwo_lines)]):
            if any([config_index < results_index < config_index_next for results_index in results_indexes]):
                results_config_indexes.append(config_index)
        image_indexes = results_config_indexes[index]
    else:
        image_indexes = all_config_indexes[index]
    pwscf_start_info = dict(((idx, None) for idx in indexes[_PW_START]))
    for image_index in image_indexes:
        if image_index in indexes[_PW_START]:
            prev_start_index = image_index
        else:
            prev_start_index = [idx for idx in indexes[_PW_START] if idx < image_index][-1]
        if pwscf_start_info[prev_start_index] is None:
            pwscf_start_info[prev_start_index] = parse_pwo_start(pwo_lines, prev_start_index)
        for next_index in all_config_indexes:
            if next_index > image_index:
                break
        else:
            next_index = len(pwo_lines)
        prev_structure = pwscf_start_info[prev_start_index]['atoms']
        if image_index in indexes[_PW_START]:
            structure = prev_structure.copy()
        else:
            if _PW_CELL in pwo_lines[image_index - 5]:
                cell, cell_alat = get_cell_parameters(pwo_lines[image_index - 5:image_index])
            else:
                cell = prev_structure.cell
                cell_alat = pwscf_start_info[prev_start_index]['alat']
            n_atoms = len(prev_structure)
            positions_card = get_atomic_positions(pwo_lines[image_index:image_index + n_atoms + 1], n_atoms=n_atoms, cell=cell, alat=cell_alat)
            symbols = [label_to_symbol(position[0]) for position in positions_card]
            positions = [position[1] for position in positions_card]
            structure = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
        energy = None
        for energy_index in indexes[_PW_TOTEN]:
            if image_index < energy_index < next_index:
                energy = float(pwo_lines[energy_index].split()[-2]) * units['Ry']
        forces = None
        for force_index in indexes[_PW_FORCE]:
            if image_index < force_index < next_index:
                if not pwo_lines[force_index + 2].strip():
                    force_index += 4
                else:
                    force_index += 2
                forces = [[float(x) for x in force_line.split()[-3:]] for force_line in pwo_lines[force_index:force_index + len(structure)]]
                forces = np.array(forces) * units['Ry'] / units['Bohr']
        stress = None
        for stress_index in indexes[_PW_STRESS]:
            if image_index < stress_index < next_index:
                sxx, sxy, sxz = pwo_lines[stress_index + 1].split()[:3]
                _, syy, syz = pwo_lines[stress_index + 2].split()[:3]
                _, _, szz = pwo_lines[stress_index + 3].split()[:3]
                stress = np.array([sxx, syy, szz, syz, sxz, sxy], dtype=float)
                stress *= -1 * units['Ry'] / units['Bohr'] ** 3
        magmoms = None
        for magmoms_index in indexes[_PW_MAGMOM]:
            if image_index < magmoms_index < next_index:
                magmoms = [float(mag_line.split()[5]) for mag_line in pwo_lines[magmoms_index + 1:magmoms_index + 1 + len(structure)]]
        efermi = None
        for fermi_index in indexes[_PW_FERMI]:
            if image_index < fermi_index < next_index:
                efermi = float(pwo_lines[fermi_index].split()[-2])
        if efermi is None:
            for ho_index in indexes[_PW_HIGHEST_OCCUPIED]:
                if image_index < ho_index < next_index:
                    efermi = float(pwo_lines[ho_index].split()[-1])
        if efermi is None:
            for holf_index in indexes[_PW_HIGHEST_OCCUPIED_LOWEST_FREE]:
                if image_index < holf_index < next_index:
                    efermi = float(pwo_lines[holf_index].split()[-2])
        ibzkpts = None
        weights = None
        kpoints_warning = 'Number of k-points >= 100: ' + "set verbosity='high' to print them."
        for kpts_index in indexes[_PW_KPTS]:
            nkpts = int(pwo_lines[kpts_index].split()[4])
            kpts_index += 2
            if pwo_lines[kpts_index].strip() == kpoints_warning:
                continue
            cell = structure.get_cell()
            alat = np.linalg.norm(cell[0])
            ibzkpts = []
            weights = []
            for i in range(nkpts):
                L = pwo_lines[kpts_index + i].split()
                weights.append(float(L[-1]))
                coord = np.array([L[-6], L[-5], L[-4].strip('),')], dtype=float)
                coord *= 2 * np.pi / alat
                coord = kpoint_convert(cell, ckpts_kv=coord)
                ibzkpts.append(coord)
            ibzkpts = np.array(ibzkpts)
            weights = np.array(weights)
        kpts = None
        kpoints_warning = 'Number of k-points >= 100: ' + "set verbosity='high' to print the bands."
        for bands_index in indexes[_PW_BANDS] + indexes[_PW_BANDSTRUCTURE]:
            if image_index < bands_index < next_index:
                bands_index += 2
                if pwo_lines[bands_index].strip() == kpoints_warning:
                    continue
                assert ibzkpts is not None
                spin, bands, eigenvalues = (0, [], [[], []])
                while True:
                    L = pwo_lines[bands_index].replace('-', ' -').split()
                    if len(L) == 0:
                        if len(bands) > 0:
                            eigenvalues[spin].append(bands)
                            bands = []
                    elif L == ['occupation', 'numbers']:
                        bands_index += len(eigenvalues[spin][0]) // 8 + 1
                    elif L[0] == 'k' and L[1].startswith('='):
                        pass
                    elif 'SPIN' in L:
                        if 'DOWN' in L:
                            spin += 1
                    else:
                        try:
                            bands.extend(map(float, L))
                        except ValueError:
                            break
                    bands_index += 1
                if spin == 1:
                    assert len(eigenvalues[0]) == len(eigenvalues[1])
                assert len(eigenvalues[0]) == len(ibzkpts), (np.shape(eigenvalues), len(ibzkpts))
                kpts = []
                for s in range(spin + 1):
                    for w, k, e in zip(weights, ibzkpts, eigenvalues[s]):
                        kpt = SinglePointKPoint(w, s, k, eps_n=e)
                        kpts.append(kpt)
        calc = SinglePointDFTCalculator(structure, energy=energy, free_energy=energy, forces=forces, stress=stress, magmoms=magmoms, efermi=efermi, ibzkpts=ibzkpts)
        calc.kpts = kpts
        structure.calc = calc
        yield structure