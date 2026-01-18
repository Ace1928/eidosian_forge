import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
def report_IC(entity: Union[Structure, Model, Chain, Residue], reportDict: Dict[str, Any]=None, verbose: bool=False) -> Dict[str, Any]:
    """Generate dict with counts of ic data elements for each entity level.

    reportDict entries are:
        - idcode : PDB ID
        - hdr : PDB header lines
        - mdl : models
        - chn : chains
        - res : residue objects
        - res_e : residues with dihedra and/or hedra
        - dih : dihedra
        - hed : hedra

    :param Entity entity: Biopython PDB Entity object: S, M, C or R
    :raises PDBException: if entity level not S, M, C, or R
    :raises Exception: if entity does not have .level attribute
    :returns: dict with counts of IC data elements
    """
    if reportDict is None:
        reportDict = {'idcode': None, 'hdr': 0, 'mdl': 0, 'chn': 0, 'chn_ids': [], 'res': 0, 'res_e': 0, 'dih': 0, 'hed': 0}
    try:
        if 'A' == entity.level:
            raise PDBException('No IC output at Atom level')
        elif isinstance(entity, (DisorderedResidue, Residue)):
            if entity.internal_coord:
                reportDict['res'] += 1
                dlen = len(entity.internal_coord.dihedra)
                hlen = len(entity.internal_coord.hedra)
                if 0 < dlen or 0 < hlen:
                    reportDict['res_e'] += 1
                    reportDict['dih'] += dlen
                    reportDict['hed'] += hlen
        elif isinstance(entity, Chain):
            reportDict['chn'] += 1
            reportDict['chn_ids'].append(entity.id)
            for res in entity:
                reportDict = report_IC(res, reportDict)
        elif isinstance(entity, Model):
            reportDict['mdl'] += 1
            for chn in entity:
                reportDict = report_IC(chn, reportDict)
        elif isinstance(entity, Structure):
            if hasattr(entity, 'header'):
                if reportDict['idcode'] is None:
                    reportDict['idcode'] = entity.header.get('idcode', None)
                hdr = entity.header.get('head', None)
                if hdr:
                    reportDict['hdr'] += 1
                nam = entity.header.get('name', None)
                if nam:
                    reportDict['hdr'] += 1
            for mdl in entity:
                reportDict = report_IC(mdl, reportDict)
        else:
            raise PDBException('Cannot identify level: ' + str(entity.level))
    except KeyError:
        raise Exception('write_PIC: argument is not a Biopython PDB Entity ' + str(entity))
    if verbose:
        print('{} : {} models {} chains {} {} residue objects {} residues with {} dihedra {} hedra'.format(reportDict['idcode'], reportDict['mdl'], reportDict['chn'], reportDict['chn_ids'], reportDict['res'], reportDict['res_e'], reportDict['dih'], reportDict['hed']))
    return reportDict