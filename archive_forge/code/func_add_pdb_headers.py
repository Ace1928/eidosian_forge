import dataclasses
import re
import string
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
from . import residue_constants
def add_pdb_headers(prot: Protein, pdb_str: str) -> str:
    """Add pdb headers to an existing PDB string. Useful during multi-chain
    recycling
    """
    out_pdb_lines: List[str] = []
    lines = pdb_str.split('\n')
    remark = prot.remark
    if remark is not None:
        out_pdb_lines.append(f'REMARK {remark}')
    parents_per_chain: List[List[str]]
    if prot.parents is not None and len(prot.parents) > 0:
        parents_per_chain = []
        if prot.parents_chain_index is not None:
            parent_dict: Dict[str, List[str]] = {}
            for p, i in zip(prot.parents, prot.parents_chain_index):
                parent_dict.setdefault(str(i), [])
                parent_dict[str(i)].append(p)
            max_idx = max([int(chain_idx) for chain_idx in parent_dict])
            for i in range(max_idx + 1):
                chain_parents = parent_dict.get(str(i), ['N/A'])
                parents_per_chain.append(chain_parents)
        else:
            parents_per_chain.append(list(prot.parents))
    else:
        parents_per_chain = [['N/A']]

    def make_parent_line(p: Sequence[str]) -> str:
        return f'PARENT {' '.join(p)}'
    out_pdb_lines.append(make_parent_line(parents_per_chain[0]))
    chain_counter = 0
    for i, l in enumerate(lines):
        if 'PARENT' not in l and 'REMARK' not in l:
            out_pdb_lines.append(l)
        if 'TER' in l and 'END' not in lines[i + 1]:
            chain_counter += 1
            if not chain_counter >= len(parents_per_chain):
                chain_parents = parents_per_chain[chain_counter]
            else:
                chain_parents = ['N/A']
            out_pdb_lines.append(make_parent_line(chain_parents))
    return '\n'.join(out_pdb_lines)