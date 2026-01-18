from __future__ import annotations
import datetime
import json
import re
import sys
from collections import namedtuple
from io import StringIO
from typing import TYPE_CHECKING
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core.structure import Molecule, Structure
def is_valid_bibtex(reference: str) -> bool:
    """Use pybtex to validate that a reference is in proper BibTeX format.

    Args:
        reference: A String reference in BibTeX format.

    Returns:
        Boolean indicating if reference is valid bibtex.
    """
    sio = StringIO(reference.encode('ascii', 'ignore').decode('ascii'))
    parser = bibtex.Parser()
    errors.set_strict_mode(enable=False)
    bib_data = parser.parse_stream(sio)
    return len(bib_data.entries) > 0