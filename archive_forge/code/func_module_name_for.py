from __future__ import annotations
import json
import os
import pathlib
import pickle
import re
import sys
import typing as T
from ..backend.ninjabackend import ninja_quote
from ..compilers.compilers import lang_suffixes
def module_name_for(self, src: str) -> str:
    suffix = os.path.splitext(src)[1][1:].lower()
    if suffix in lang_suffixes['fortran']:
        exported = self.exports[src]
        namebase = exported.replace(':', '@')
        if ':' in exported:
            extension = 'smod'
        else:
            extension = 'mod'
        return os.path.join(self.target_data.private_dir, f'{namebase}.{extension}')
    elif suffix in lang_suffixes['cpp']:
        return '{}.ifc'.format(self.exports[src])
    else:
        raise RuntimeError('Unreachable code.')