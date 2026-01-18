from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
from fontTools.designspaceLib import (
def regionInRegion(region: Region, superRegion: Region) -> bool:
    for name, value in region.items():
        if not name in superRegion:
            return False
        superValue = superRegion[name]
        if isinstance(superValue, (float, int)):
            if value != superValue:
                return False
        elif value not in superValue:
            return False
    return True