import re
from typing import Optional, Type
from . import errors, hooks, registry, urlutils
def get_forge_by_hostname(hostname: str):
    """Get a forge from a hostname.
    """
    for instance in iter_forge_instances():
        try:
            return instance.probe_from_hostname(hostname)
        except UnsupportedForge:
            pass
    raise UnsupportedForge(hostname)