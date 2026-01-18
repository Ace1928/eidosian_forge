from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def get_header_chunk(content: str | TextIOWrapper) -> AimsOutHeaderChunk:
    """Get the header chunk for an output

    Args:
        content (str or TextIOWrapper): the content to parse

    Returns:
        The AimsHeaderChunk of the file
    """
    lines = get_lines(content)
    header = []
    stopped = False
    for line in lines:
        header.append(line)
        if 'Convergence:    q app. |  density  | eigen (eV) | Etot (eV)' in line or 'Begin self-consistency iteration #' in line:
            stopped = True
            break
    if not stopped:
        raise ParseError('No SCF steps present, calculation failed at setup.')
    return AimsOutHeaderChunk(header)