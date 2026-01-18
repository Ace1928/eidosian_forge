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
def reorder_blend_args(self, commands, get_delta_func):
    """
        We first re-order the master coordinate values.
        For a moveto to lineto, the args are now arranged as::

                [ [master_0 x,y], [master_1 x,y], [master_2 x,y] ]

        We re-arrange this to::

                [	[master_0 x, master_1 x, master_2 x],
                        [master_0 y, master_1 y, master_2 y]
                ]

        If the master values are all the same, we collapse the list to
        as single value instead of a list.

        We then convert this to::

                [ [master_0 x] + [x delta tuple] + [numBlends=1]
                  [master_0 y] + [y delta tuple] + [numBlends=1]
                ]
        """
    for cmd in commands:
        args = cmd[1]
        m_args = zip(*args)
        cmd[1] = list(m_args)
    lastOp = None
    for cmd in commands:
        op = cmd[0]
        if lastOp in ['hintmask', 'cntrmask']:
            coord = list(cmd[1])
            if not allEqual(coord):
                raise VarLibMergeError('Hintmask values cannot differ between source fonts.')
            cmd[1] = [coord[0][0]]
        else:
            coords = cmd[1]
            new_coords = []
            for coord in coords:
                if allEqual(coord):
                    new_coords.append(coord[0])
                else:
                    deltas = get_delta_func(coord)[1:]
                    coord = [coord[0]] + deltas
                    coord.append(1)
                    new_coords.append(coord)
            cmd[1] = new_coords
        lastOp = op
    return commands