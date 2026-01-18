from __future__ import annotations
from dataclasses import dataclass
import subprocess
import typing as T
from enum import Enum
from . import mesonlib
from .mesonlib import EnvironmentException, HoldableObject
from . import mlog
from pathlib import Path
@classmethod
def from_literal(cls, literal: T.Dict[str, str]) -> 'MachineInfo':
    minimum_literal = {'cpu', 'cpu_family', 'endian', 'system'}
    if set(literal) < minimum_literal:
        raise EnvironmentException(f'Machine info is currently {literal}\n' + 'but is missing {}.'.format(minimum_literal - set(literal)))
    cpu_family = literal['cpu_family']
    if cpu_family not in known_cpu_families:
        mlog.warning(f'Unknown CPU family {cpu_family}, please report this at https://github.com/mesonbuild/meson/issues/new')
    endian = literal['endian']
    if endian not in ('little', 'big'):
        mlog.warning(f'Unknown endian {endian}')
    system = literal['system']
    kernel = literal.get('kernel', None)
    subsystem = literal.get('subsystem', None)
    return cls(system, cpu_family, literal['cpu'], endian, kernel, subsystem)