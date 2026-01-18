from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def get_aims_out_chunks(content: str | TextIOWrapper, header_chunk: AimsOutHeaderChunk) -> Generator:
    """Yield unprocessed chunks (header, lines) for each AimsOutChunk image.

    Args:
        content (str or TextIOWrapper): the content to parse
        header_chunk (AimsOutHeaderChunk): The AimsOutHeader for the calculation

    Yields:
        The next AimsOutChunk
    """
    lines = get_lines(content)[len(header_chunk.lines):]
    if len(lines) == 0:
        return
    if header_chunk.is_relaxation:
        chunk_end_line = 'Geometry optimization: Attempting to predict improved coordinates.'
    else:
        chunk_end_line = 'Begin self-consistency loop: Re-initialization'
    ignore_chunk_end_line = False
    line_iter = iter(lines)
    while True:
        try:
            line = next(line_iter).strip()
        except StopIteration:
            break
        chunk_lines = []
        while chunk_end_line not in line or ignore_chunk_end_line:
            chunk_lines.append(line)
            patterns = ['Self-consistency cycle not yet converged - restarting mixer to attempt better convergence.', 'Components of the stress tensor (for mathematical background see comments in numerical_stress.f90).', 'Calculation of numerical stress completed']
            if any((pattern in line for pattern in patterns)):
                ignore_chunk_end_line = True
            elif 'Begin self-consistency loop: Re-initialization' in line:
                ignore_chunk_end_line = False
            try:
                line = next(line_iter).strip()
            except StopIteration:
                break
        yield AimsOutCalcChunk(chunk_lines, header_chunk)