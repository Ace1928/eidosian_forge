from __future__ import annotations
import functools, uuid, os, operator
import typing as T
from . import backends
from .. import build
from .. import mesonlib
from .. import mlog
from ..mesonlib import MesonBugException, MesonException, OptionKey
def create_generator_shellphase(self, objects_dict, tname, generator_id) -> None:
    file_ids = self.generator_buildfile_ids[tname, generator_id]
    ref_ids = self.generator_fileref_ids[tname, generator_id]
    assert len(ref_ids) == len(file_ids)
    for file_o, ref_id in zip(file_ids, ref_ids):
        odict = PbxDict()
        objects_dict.add_item(file_o, odict)
        odict.add_item('isa', 'PBXBuildFile')
        odict.add_item('fileRef', ref_id)