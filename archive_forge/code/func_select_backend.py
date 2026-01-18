from importlib import import_module
from typing import Dict
def select_backend(name: str) -> Dict:
    """Select the pyzmq backend"""
    try:
        mod = import_module(name)
    except ImportError:
        raise
    except Exception as e:
        raise ImportError(f'Importing {name} failed with {e}') from e
    ns = {'monitored_queue': mod.monitored_queue}
    ns.update({key: getattr(mod, key) for key in public_api})
    return ns