from io import BytesIO
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from ase.io.formats import ioformats
from ase.io import write
def sage(self, atoms):
    from ase.visualize.sage import view_sage_jmol
    return view_sage_jmol(atoms)