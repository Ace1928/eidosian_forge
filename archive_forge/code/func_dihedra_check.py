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
def dihedra_check(ric: IC_Residue) -> None:
    """Look for required dihedra in residue, generate defaults if set."""

    def ake_recurse(akList: List) -> List:
        """Bulid combinatorics of AtomKey lists."""
        car = akList[0]
        if len(akList) > 1:
            retList = []
            for ak in car:
                cdr = akList[1:]
                rslt = ake_recurse(cdr)
                for r in rslt:
                    r.insert(0, ak)
                    retList.append(r)
            return retList
        elif len(car) == 1:
            return [list(car)]
        else:
            retList = [[ak] for ak in car]
            return retList

    def ak_expand(eLst: List) -> List:
        """Expand AtomKey list with altlocs, all combinatorics."""
        retList = []
        for edron in eLst:
            newList = []
            for ak in edron:
                rslt = ak.ric.split_akl([ak])
                rlst = [r[0] for r in rslt]
                if rlst != []:
                    newList.append(rlst)
                else:
                    newList.append([ak])
            rslt = ake_recurse(newList)
            for r in rslt:
                retList.append(r)
        return retList
    chkLst = []
    sN, sCA, sC = (AtomKey(ric, 'N'), AtomKey(ric, 'CA'), AtomKey(ric, 'C'))
    sO, sCB, sH = (AtomKey(ric, 'O'), AtomKey(ric, 'CB'), AtomKey(ric, 'H'))
    if ric.rnext != []:
        for rn in ric.rnext:
            nN, nCA, nC = (AtomKey(rn, 'N'), AtomKey(rn, 'CA'), AtomKey(rn, 'C'))
            chkLst.append((sN, sCA, sC, nN))
            chkLst.append((sCA, sC, nN, nCA))
            chkLst.append((sC, nN, nCA, nC))
    else:
        chkLst.append((sN, sCA, sC, AtomKey(ric, 'OXT')))
        rn = '(no rnext)'
    chkLst.append((sN, sCA, sC, sO))
    if ric.lc != 'G':
        chkLst.append((sO, sC, sCA, sCB))
        if ric.lc == 'A':
            chkLst.append((sN, sCA, sCB))
    if ric.rprev != [] and ric.lc != 'P' and proton:
        chkLst.append((sC, sCA, sN, sH))
    try:
        for edron in ic_data_sidechains[ric.lc]:
            if len(edron) > 3:
                if all((atm[0] != 'H' for atm in edron)):
                    akl = [AtomKey(ric, atm) for atm in edron[0:4]]
                    chkLst.append(akl)
    except KeyError:
        pass
    chkLst = ak_expand(chkLst)
    altloc_ndx = AtomKey.fields.altloc
    for dk in chkLst:
        if tuple(dk) in ric.dihedra:
            pass
        elif sH in dk:
            pass
        elif all((atm.akl[altloc_ndx] is None for atm in dk)):
            if defaults:
                if len(dk) != 3:
                    default_dihedron(dk, ric)
                else:
                    default_hedron(dk, ric)
            elif verbose:
                print(f'{ric}-{rn} missing {dk}')
        else:
            pass