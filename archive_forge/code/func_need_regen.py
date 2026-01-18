from __future__ import annotations
import sys, os
import pickle, subprocess
import typing as T
from ..coredata import CoreData
from ..backend.backends import RegenInfo
from ..mesonlib import OptionKey
def need_regen(regeninfo: RegenInfo, regen_timestamp: float) -> bool:
    for i in regeninfo.depfiles:
        curfile = os.path.join(regeninfo.build_dir, i)
        curtime = os.stat(curfile).st_mtime
        if curtime > regen_timestamp:
            return True
    print('Everything is up-to-date, regeneration of build files is not needed.')
    from ..backend.vs2010backend import Vs2010Backend
    Vs2010Backend.touch_regen_timestamp(regeninfo.build_dir)
    return False