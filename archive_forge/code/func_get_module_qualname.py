import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def get_module_qualname(self, file_path: Path, top_package_path: Path) -> List[str]:
    normalized_path = file_path.relative_to(top_package_path.parent)
    if normalized_path.name == '__init__.py':
        module_basename = normalized_path.parent.name
        module_parent = normalized_path.parent.parent.parts
    else:
        module_basename = normalized_path.stem
        module_parent = normalized_path.parent.parts
    return list(module_parent) + [module_basename]