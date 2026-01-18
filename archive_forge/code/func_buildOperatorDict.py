from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def buildOperatorDict(operatorList):
    oper = {}
    opc = {}
    for item in operatorList:
        if len(item) == 2:
            oper[item[0]] = item[1]
        else:
            oper[item[0]] = item[1:]
        if isinstance(item[0], tuple):
            opc[item[1]] = item[0]
        else:
            opc[item[1]] = (item[0],)
    return (oper, opc)