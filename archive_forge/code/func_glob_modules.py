from pathlib import Path
from importlib import import_module
from numpy import VisibleDeprecationWarning
import pytest
import ase
def glob_modules():
    topdir = Path(ase.__file__).parent
    for path in topdir.rglob('*.py'):
        path = 'ase' / path.relative_to(topdir)
        if path.name.startswith('__'):
            continue
        if path.parts[1] == 'test':
            continue
        yield path