from __future__ import annotations
import dataclasses
import typing as T
from .. import build, mesonlib
from ..build import IncludeDirs
from ..interpreterbase.decorators import noKwargs, noPosargs
from ..mesonlib import relpath, HoldableObject, MachineChoice
from ..programs import ExternalProgram
def find_tool(self, name: str, depname: str, varname: str, required: bool=True, wanted: T.Optional[str]=None) -> T.Union['build.Executable', ExternalProgram, 'OverrideProgram']:
    progobj = self._interpreter.program_from_overrides([name], [])
    if progobj is not None:
        return progobj
    prog_list = self.environment.lookup_binary_entry(MachineChoice.HOST, name)
    if prog_list is not None:
        return ExternalProgram.from_entry(name, prog_list)
    dep = self.dependency(depname, native=True, required=False, wanted=wanted)
    if dep.found() and dep.type_name == 'pkgconfig':
        value = dep.get_variable(pkgconfig=varname)
        if value:
            progobj = ExternalProgram(value)
            if not progobj.found():
                msg = f'Dependency {depname!r} tool variable {varname!r} contains erroneous value: {value!r}\n\nThis is a distributor issue -- please report it to your {depname} provider.'
                raise mesonlib.MesonException(msg)
            return progobj
    return self.find_program(name, required=required, wanted=wanted)