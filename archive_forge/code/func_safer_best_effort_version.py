import re
from pathlib import Path
from typing import Union
from .extern import packaging
def safer_best_effort_version(value: str) -> str:
    """Like ``best_effort_version`` but can be used as filename component for wheel"""
    return filename_component(best_effort_version(value))