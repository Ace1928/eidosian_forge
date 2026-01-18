import importlib
from typing import List, Tuple
from ase.utils import search_current_git_hash
def format_dependency(modname: str) -> Tuple[str, str]:
    """Return (name, path) for given module."""
    try:
        module = importlib.import_module(modname)
    except ImportError:
        return (modname, 'not installed')
    version = getattr(module, '__version__', '?')
    name = f'{modname}-{version}'
    if modname == 'ase':
        githash = search_current_git_hash(module)
        if githash:
            name += '-{:.10}'.format(githash)
    return (name, str(module.__path__[0]))