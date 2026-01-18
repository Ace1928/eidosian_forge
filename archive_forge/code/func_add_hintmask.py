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
def add_hintmask(self, hint_type, abs_args):
    if self.m_index == 0:
        self._commands.append([hint_type, []])
        self._commands.append(['', [abs_args]])
    else:
        cmd = self._commands[self.pt_index]
        if cmd[0] != hint_type:
            raise VarLibCFFHintTypeMergeError(hint_type, self.pt_index, len(cmd[1]), cmd[0], self.glyphName)
        self.pt_index += 1
        cmd = self._commands[self.pt_index]
        cmd[1].append(abs_args)
    self.pt_index += 1