from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def gen_single_target_map(self, genlist, tname, t, generator_id) -> None:
    k = (tname, generator_id)
    assert k not in self.shell_targets
    self.shell_targets[k] = self.gen_id()
    ofile_abs = []
    for i in genlist.get_inputs():
        for o_base in genlist.get_outputs_for(i):
            o = os.path.join(self.get_target_private_dir(t), o_base)
            ofile_abs.append(os.path.join(self.environment.get_build_dir(), o))
    assert k not in self.generator_outputs
    self.generator_outputs[k] = ofile_abs
    buildfile_ids = []
    fileref_ids = []
    for i in range(len(ofile_abs)):
        buildfile_ids.append(self.gen_id())
        fileref_ids.append(self.gen_id())
    self.generator_buildfile_ids[k] = buildfile_ids
    self.generator_fileref_ids[k] = fileref_ids