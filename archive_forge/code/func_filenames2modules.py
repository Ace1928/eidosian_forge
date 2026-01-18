from pathlib import Path
from importlib import import_module
from numpy import VisibleDeprecationWarning
import pytest
import ase
def filenames2modules(filenames):
    modules = []
    for filename in filenames:
        filename = Path(filename).as_posix()
        module = filename.rsplit('.', 1)[0]
        module = module.replace('/', '.')
        if module == 'ase.data.tmxr200x':
            continue
        modules.append(module)
        print(module)
    return modules