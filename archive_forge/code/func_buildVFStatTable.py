from __future__ import annotations
from typing import Dict, List, Union
import fontTools.otlLib.builder
from fontTools.designspaceLib import (
from fontTools.designspaceLib.types import Region, getVFUserRegion, locationInRegion
from fontTools.ttLib import TTFont
def buildVFStatTable(ttFont: TTFont, doc: DesignSpaceDocument, vfName: str) -> None:
    """Build the STAT table for the variable font identified by its name in
    the given document.

    Knowing which variable we're building STAT data for is needed to subset
    the STAT locations to only include what the variable font actually ships.

    .. versionadded:: 5.0

    .. seealso::
        - :func:`getStatAxes()`
        - :func:`getStatLocations()`
        - :func:`fontTools.otlLib.builder.buildStatTable()`
    """
    for vf in doc.getVariableFonts():
        if vf.name == vfName:
            break
    else:
        raise DesignSpaceDocumentError(f'Cannot find the variable font by name {vfName}')
    region = getVFUserRegion(doc, vf)
    return fontTools.otlLib.builder.buildStatTable(ttFont, getStatAxes(doc, region), getStatLocations(doc, region), doc.elidedFallbackName if doc.elidedFallbackName is not None else 2)