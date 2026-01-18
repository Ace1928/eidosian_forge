import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def select_fragments(fragments, ftype, hac):
    if ftype == FTYPE_ACYCLIC:
        result = []
        result_hcount = 0
        for fMol in fragments:
            nAttachments = len(fMol.GetAtomsMatchingQuery(dummyAtomQuery))
            if nAttachments == 1:
                fhac = fMol.GetNumAtoms()
                if fhac > 3:
                    result.append(Chem.MolToSmiles(fMol))
                    result_hcount += fhac
        if result and result_hcount > 0.6 * hac:
            return '.'.join(result)
        return None
    if ftype == FTYPE_CYCLIC:
        if len(fragments) != 2:
            return None
        result = None
        for fMol in fragments:
            f = Chem.MolToSmiles(fMol)
            if isValidRingCut(fMol):
                result_hcount = fMol.GetNumAtoms()
                if result_hcount > 3 and result_hcount > 0.4 * hac:
                    result = f
        return result
    if ftype == FTYPE_CYCLIC_ACYCLIC:
        result = []
        result_hcount = 0
        for fMol in fragments:
            nAttachments = len(fMol.GetAtomsMatchingQuery(dummyAtomQuery))
            if nAttachments >= 3:
                continue
            fhac = fMol.GetNumAtoms()
            if fhac <= 3:
                continue
            if nAttachments == 2:
                if isValidRingCut(fMol):
                    result.append(Chem.MolToSmiles(fMol))
                    result_hcount += fhac
            elif nAttachments == 1:
                result.append(Chem.MolToSmiles(fMol))
                result_hcount += fhac
        if len(result) == 2 and result_hcount > 0.6 * hac:
            return '.'.join(result)
        return None
    raise NotImplementedError(f'Invalid fragmentation type {type}')