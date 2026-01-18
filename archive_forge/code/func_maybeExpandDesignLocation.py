from __future__ import annotations
import itertools
import logging
import math
from typing import Any, Callable, Dict, Iterator, List, Tuple, cast
from fontTools.designspaceLib import (
from fontTools.designspaceLib.statNames import StatNames, getStatNames
from fontTools.designspaceLib.types import (
def maybeExpandDesignLocation(object):
    if expandLocations:
        return object.getFullDesignLocation(doc)
    else:
        return object.designLocation