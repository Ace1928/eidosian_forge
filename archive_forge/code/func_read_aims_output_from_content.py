from __future__ import annotations
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.core.tensors import Tensor
def read_aims_output_from_content(content: str, index: int | slice=-1, non_convergence_ok: bool=False) -> Structure | Molecule | Sequence[Structure | Molecule]:
    """Read and aims output file from the content of a file

    Args:
      content (str): The content of the file to read
      index: int | slice:  (Default value = -1)
      non_convergence_ok: bool:  (Default value = False)

    Returns:
        The list of images to get
    """
    header_chunk = get_header_chunk(content)
    chunks = list(get_aims_out_chunks(content, header_chunk))
    check_convergence(chunks, non_convergence_ok)
    images = [chunk.structure for chunk in chunks[:-1]] if header_chunk.is_relaxation else [chunk.structure for chunk in chunks]
    return images[index]