import os
from kombu.utils.imports import symbol_by_name
def get_available_pool_names():
    """Return all available pool type names."""
    return tuple(ALIASES.keys())