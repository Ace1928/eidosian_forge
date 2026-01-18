from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def splitVariableFonts(doc: DesignSpaceDocument, makeNames: bool=False, expandLocations: bool=False, makeInstanceFilename: MakeInstanceFilenameCallable=defaultMakeInstanceFilename) -> Iterator[Tuple[str, DesignSpaceDocument]]:
    """Convert each variable font listed in this document into a standalone
    designspace. This can be used to compile all the variable fonts from a
    format 5 designspace using tools that can only deal with 1 VF at a time.

    Args:
      - ``makeNames``: Whether to compute the instance family and style
        names using the STAT data.
      - ``expandLocations``: Whether to turn all locations into "full"
        locations, including implicit default axis values where missing.
      - ``makeInstanceFilename``: Callable to synthesize an instance filename
        when makeNames=True, for instances that don't specify an instance name
        in the designspace. This part of the name generation can be overridden
        because it's not specified by the STAT table.

    .. versionadded:: 5.0
    """
    for vf in doc.getVariableFonts():
        vfUserRegion = getVFUserRegion(doc, vf)
        vfDoc = _extractSubSpace(doc, vfUserRegion, keepVFs=False, makeNames=makeNames, expandLocations=expandLocations, makeInstanceFilename=makeInstanceFilename)
        vfDoc.lib = {**vfDoc.lib, **vf.lib}
        yield (vf.name, vfDoc)