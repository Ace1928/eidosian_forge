from __future__ import annotations
from .common import cmake_is_debug
from .. import mlog
from ..mesonlib import Version
from pathlib import Path
import re
import typing as T
def resolve_cmake_trace_targets(target_name: str, trace: 'CMakeTraceParser', env: 'Environment', *, clib_compiler: T.Union['MissingCompiler', 'Compiler']=None, not_found_warning: T.Callable[[str], None]=lambda x: None) -> ResolvedTarget:
    res = ResolvedTarget()
    targets = [target_name]
    reg_is_lib = re.compile('^(-l[a-zA-Z0-9_]+|-l?pthread)$')
    reg_is_maybe_bare_lib = re.compile('^[a-zA-Z0-9_]+$')
    is_debug = cmake_is_debug(env)
    processed_targets: T.List[str] = []
    while len(targets) > 0:
        curr = targets.pop(0)
        if curr in processed_targets:
            continue
        if curr not in trace.targets:
            curr_path = Path(curr)
            if reg_is_lib.match(curr):
                res.libraries += [curr]
            elif curr_path.is_absolute() and curr_path.exists():
                if any((x.endswith('.framework') for x in curr_path.parts)):
                    path_to_framework = []
                    for x in curr_path.parts:
                        path_to_framework.append(x)
                        if x.endswith('.framework'):
                            break
                    curr_path = Path(*path_to_framework)
                    framework_path = curr_path.parent
                    framework_name = curr_path.stem
                    res.libraries += [f'-F{framework_path}', '-framework', framework_name]
                else:
                    res.libraries += [curr]
            elif reg_is_maybe_bare_lib.match(curr) and clib_compiler:
                flib = clib_compiler.find_library(curr, env, [])
                if flib is not None:
                    res.libraries += flib
                else:
                    not_found_warning(curr)
            else:
                not_found_warning(curr)
            continue
        tgt = trace.targets[curr]
        cfgs = []
        cfg = ''
        mlog.debug(tgt)
        if 'INTERFACE_INCLUDE_DIRECTORIES' in tgt.properties:
            res.include_directories += [x for x in tgt.properties['INTERFACE_INCLUDE_DIRECTORIES'] if x]
        if 'INTERFACE_LINK_OPTIONS' in tgt.properties:
            res.link_flags += [x for x in tgt.properties['INTERFACE_LINK_OPTIONS'] if x]
        if 'INTERFACE_COMPILE_DEFINITIONS' in tgt.properties:
            res.public_compile_opts += ['-D' + re.sub('^-D', '', x) for x in tgt.properties['INTERFACE_COMPILE_DEFINITIONS'] if x]
        if 'INTERFACE_COMPILE_OPTIONS' in tgt.properties:
            res.public_compile_opts += [x for x in tgt.properties['INTERFACE_COMPILE_OPTIONS'] if x]
        if 'IMPORTED_CONFIGURATIONS' in tgt.properties:
            cfgs = [x for x in tgt.properties['IMPORTED_CONFIGURATIONS'] if x]
            cfg = cfgs[0]
        if is_debug:
            if 'DEBUG' in cfgs:
                cfg = 'DEBUG'
            elif 'RELEASE' in cfgs:
                cfg = 'RELEASE'
        elif 'RELEASE' in cfgs:
            cfg = 'RELEASE'
        if f'IMPORTED_IMPLIB_{cfg}' in tgt.properties:
            res.libraries += [x for x in tgt.properties[f'IMPORTED_IMPLIB_{cfg}'] if x]
        elif 'IMPORTED_IMPLIB' in tgt.properties:
            res.libraries += [x for x in tgt.properties['IMPORTED_IMPLIB'] if x]
        elif f'IMPORTED_LOCATION_{cfg}' in tgt.properties:
            res.libraries += [x for x in tgt.properties[f'IMPORTED_LOCATION_{cfg}'] if x]
        elif 'IMPORTED_LOCATION' in tgt.properties:
            res.libraries += [x for x in tgt.properties['IMPORTED_LOCATION'] if x]
        if 'LINK_LIBRARIES' in tgt.properties:
            targets += [x for x in tgt.properties['LINK_LIBRARIES'] if x]
        if 'INTERFACE_LINK_LIBRARIES' in tgt.properties:
            targets += [x for x in tgt.properties['INTERFACE_LINK_LIBRARIES'] if x]
        if f'IMPORTED_LINK_DEPENDENT_LIBRARIES_{cfg}' in tgt.properties:
            targets += [x for x in tgt.properties[f'IMPORTED_LINK_DEPENDENT_LIBRARIES_{cfg}'] if x]
        elif 'IMPORTED_LINK_DEPENDENT_LIBRARIES' in tgt.properties:
            targets += [x for x in tgt.properties['IMPORTED_LINK_DEPENDENT_LIBRARIES'] if x]
        processed_targets += [curr]
    return res