import re
from datetime import date
from io import StringIO
import numpy as np
from Bio.File import as_handle
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB.parse_pdb_header import _parse_pdb_header_list
from Bio.PDB.PDBExceptions import PDBException
from Bio.Data.PDBData import protein_letters_1to3
from Bio.PDB.internal_coords import (
from Bio.PDB.ic_data import (
from typing import TextIO, Set, List, Tuple, Union, Optional
from Bio.PDB.Structure import Structure
from Bio.PDB.Residue import Residue
from Bio import SeqIO
def default_dihedron(ek: List, ric: IC_Residue) -> None:
    """Create Dihedron based on same residue class dihedra in ref database.

        Adds Dihedron to current Chain.internal_coord, see ic_data for default
        values and reference database source.
        """
    atmNdx = AtomKey.fields.atm
    resNdx = AtomKey.fields.resname
    resPos = AtomKey.fields.respos
    rdclass = ''
    dclass = ''
    for ak in ek:
        dclass += ak.akl[atmNdx]
        rdclass += ak.akl[resNdx] + ak.akl[atmNdx]
    if dclass == 'NCACN':
        rdclass = rdclass[0:7] + 'XN'
    elif dclass == 'CACNCA':
        rdclass = 'XCAXC' + rdclass[5:]
    elif dclass == 'CNCAC':
        rdclass = 'XC' + rdclass[2:]
    if rdclass in dihedra_primary_defaults:
        process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), dihedra_primary_defaults[rdclass][0], ric)
        if verbose:
            print(f' default for {ek}')
    elif rdclass in dihedra_secondary_defaults:
        primAngle, offset = dihedra_secondary_defaults[rdclass]
        rname = ek[2].akl[resNdx]
        rnum = int(ek[2].akl[resPos])
        paKey = None
        if primAngle == ('N', 'CA', 'C', 'N') and ek[0].ric.rnext != []:
            paKey = [AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(3)]
            rnext = ek[0].ric.rnext
            paKey.append(AtomKey((rnext[0].rbase[0], None, rnext[0].rbase[2], 'N', None, None)))
            paKey = tuple(paKey)
        elif primAngle == ('CA', 'C', 'N', 'CA'):
            prname = pr.akl[0][resNdx]
            prnum = pr.akl[0][resPos]
            paKey = [AtomKey(prnum, None, prname, primAngle[x], None, None) for x in range(2)]
            paKey.add([AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(2, 4)])
            paKey = tuple(paKey)
        else:
            paKey = tuple((AtomKey((rnum, None, rname, atm, None, None)) for atm in primAngle))
        if paKey in da:
            angl = da[paKey] + dihedra_secondary_defaults[rdclass][1]
            process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), angl, ric)
            if verbose:
                print(f' secondary default for {ek}')
        elif rdclass in dihedra_secondary_xoxt_defaults:
            if primAngle == ('C', 'N', 'CA', 'C'):
                prname = pr.akl[0][resNdx]
                prnum = pr.akl[0][resPos]
                paKey = [AtomKey(prnum, None, prname, primAngle[0], None, None)]
                paKey.add([AtomKey((rnum, None, rname, primAngle[x], None, None)) for x in range(1, 4)])
                paKey = tuple(paKey)
            else:
                primAngle, offset = dihedra_secondary_xoxt_defaults[rdclass]
                rname = ek[2].akl[resNdx]
                rnum = int(ek[2].akl[resPos])
                paKey = tuple((AtomKey((rnum, None, rname, atm, None, None)) for atm in primAngle))
            if paKey in da:
                angl = da[paKey] + offset
                process_dihedron(str(ek[0]), str(ek[1]), str(ek[2]), str(ek[3]), angl, ric)
                if verbose:
                    print(f' oxt default for {ek}')
            else:
                print(f'missing primary angle {paKey} {primAngle} to generate {rnum}{rname} {rdclass}')
    else:
        print(f'missing {ek} -> {rdclass} ({dclass}) not found in primary or secondary defaults')