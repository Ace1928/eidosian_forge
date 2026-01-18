import os
import shutil
import importlib
from ase.calculators.calculator import names
def detect_calculators():
    configs = {}
    for name in names:
        result = detect(name)
        if result:
            configs[name] = result
    return configs