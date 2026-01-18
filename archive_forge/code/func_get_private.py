from collections import namedtuple
from fontTools.cffLib import (
from io import BytesIO
from fontTools.cffLib.specializer import specializeCommands, commandsToProgram
from fontTools.ttLib import newTable
from fontTools import varLib
from fontTools.varLib.models import allEqual
from fontTools.misc.roundTools import roundFunc
from fontTools.misc.psCharStrings import T2CharString, T2OutlineExtractor
from fontTools.pens.t2CharStringPen import T2CharStringPen
from functools import partial
from .errors import (
def get_private(regionFDArrays, fd_index, ri, fd_map):
    region_fdArray = regionFDArrays[ri]
    region_fd_map = fd_map[fd_index]
    if ri in region_fd_map:
        region_fdIndex = region_fd_map[ri]
        private = region_fdArray[region_fdIndex].Private
    else:
        private = None
    return private