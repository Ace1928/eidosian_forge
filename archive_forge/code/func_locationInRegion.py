from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, cast
from fontTools.designspaceLib import (
def locationInRegion(location: SimpleLocationDict, region: Region) -> bool:
    for name, value in location.items():
        if name not in region:
            return False
        regionValue = region[name]
        if isinstance(regionValue, (float, int)):
            if value != regionValue:
                return False
        elif value not in regionValue:
            return False
    return True