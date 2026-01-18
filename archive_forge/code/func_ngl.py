from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
def ngl(self, atoms):
    from ase.visualize.ngl import view_ngl
    return view_ngl(atoms)