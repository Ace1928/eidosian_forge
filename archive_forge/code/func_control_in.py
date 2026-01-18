from __future__ import annotations
import copy
import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from warnings import warn
import numpy as np
from monty.json import MontyDecoder, MontyEncoder
from pymatgen.core import Molecule, Structure
from pymatgen.io.aims.inputs import AimsControlIn, AimsGeometryIn
from pymatgen.io.aims.parsers import AimsParseError, read_aims_output
from pymatgen.io.core import InputFile, InputGenerator, InputSet
@property
def control_in(self) -> str | slice | InputFile:
    """Get the control.in file contents."""
    return self[CONTROL_FILE_NAME]