from pathlib import Path
from typing import Union, Dict, Any, List, Tuple
from collections import OrderedDict
def force_path(location, require_exists=True):
    if not isinstance(location, Path):
        location = Path(location)
    if require_exists and (not location.exists()):
        raise ValueError(f"Can't read file: {location}")
    return location