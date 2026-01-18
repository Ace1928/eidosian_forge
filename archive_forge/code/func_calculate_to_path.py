import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
def calculate_to_path(p):
    if p is None:
        return p
    p = str(p)
    for from_, to in renames:
        if p.startswith(str(from_)):
            p = str(to) + p[len(str(from_)):]
    return Path(p)