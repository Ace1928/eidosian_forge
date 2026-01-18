from __future__ import annotations
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import nprint, dbg, DBG_EVENT, \
def seq_seq(self) -> bool:
    try:
        if self.values[-2][1] and self.values[-1][1]:
            return True
    except IndexError:
        pass
    return False