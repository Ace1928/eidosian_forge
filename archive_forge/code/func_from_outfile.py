from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from monty.json import MontyDecoder, MSONable
from pymatgen.io.aims.parsers import (
@classmethod
def from_outfile(cls, outfile: str | Path) -> Self:
    """Construct an AimsOutput from an output file.

        Args:
            outfile: str | Path: The aims.out file to parse

        Returns:
            The AimsOutput object for the output file
        """
    metadata, structure_summary = read_aims_header_info(outfile)
    results = read_aims_output(outfile, index=slice(0, None))
    return cls(results, metadata, structure_summary)