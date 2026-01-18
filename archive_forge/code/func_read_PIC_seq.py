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
def read_PIC_seq(seqRec: 'SeqIO.SeqRecord', pdbid: str=None, title: str=None, chain: str=None) -> Structure:
    """Read :class:`.SeqRecord` into Structure with default internal coords."""
    read_pdbid, read_title, read_chain = (None, None, None)
    if seqRec.id is not None:
        read_pdbid = seqRec.id
    if seqRec.description is not None:
        read_title = seqRec.description.replace(f'{read_pdbid} ', '')
    if ':' in read_pdbid:
        read_pdbid, read_chain = read_pdbid.split(':')
    if chain is None:
        chain = read_chain if read_chain is not None else 'A'
    if title is None:
        title = read_title if read_title is not None else f'sequence input {(seqRec.id if seqRec.id is not None else '')}'
    if pdbid is None:
        pdbid = read_pdbid if read_pdbid is not None else '0PDB'
    today = date.today()
    datestr = today.strftime('%d-%b-%y').upper()
    output = f'HEADER    {'GENERATED STRUCTURE':40}{datestr}   {pdbid}\n'
    output += f'TITLE     {title.upper():69}\n'
    ndx = 1
    for r in seqRec.seq:
        output += f"('{pdbid}', 0, '{chain}', (' ', {ndx}, ' ')) {protein_letters_1to3[r]}\n"
        ndx += 1
    sp = StringIO()
    sp.write(output)
    sp.seek(0)
    return read_PIC(sp, defaults=True)