import io
import math
import os
import typing
import weakref
def getColorInfoDict() -> dict:
    d = {}
    for item in getColorInfoList():
        d[item[0].lower()] = item[1:]
    return d