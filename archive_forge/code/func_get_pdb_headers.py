import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def get_pdb_headers(prot: Protein, chain_id: int=0) -> List[str]:
    pdb_headers: List[str] = []
    remark = prot.remark
    if remark is not None:
        pdb_headers.append(f'REMARK {remark}')
    parents = prot.parents
    parents_chain_index = prot.parents_chain_index
    if parents is not None and parents_chain_index is not None:
        parents = [p for i, p in zip(parents_chain_index, parents) if i == chain_id]
    if parents is None or len(parents) == 0:
        parents = ['N/A']
    pdb_headers.append(f'PARENT {' '.join(parents)}')
    return pdb_headers