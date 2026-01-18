from __future__ import annotations
from functools import lru_cache
from os import environ
from pathlib import Path
import re
import typing as T
from .common import CMakeException, CMakeTarget, language_map, cmake_get_generator_args, check_cmake_args
from .fileapi import CMakeFileAPI
from .executor import CMakeExecutor
from .toolchain import CMakeToolchain, CMakeExecScope
from .traceparser import CMakeTraceParser
from .tracetargets import resolve_cmake_trace_targets
from .. import mlog, mesonlib
from ..mesonlib import MachineChoice, OrderedSet, path_is_in_root, relative_to_if_possible, OptionKey
from ..mesondata import DataFile
from ..compilers.compilers import assembler_suffixes, lang_suffixes, header_suffixes, obj_suffixes, lib_suffixes, is_header
from ..programs import ExternalProgram
from ..coredata import FORBIDDEN_TARGET_NAMES
from ..mparser import (
def pretend_to_be_meson(self, options: TargetOptions) -> CodeBlockNode:
    if not self.project_name:
        raise CMakeException('CMakeInterpreter was not analysed')

    def token(tid: str='string', val: TYPE_mixed='') -> Token:
        return Token(tid, self.subdir.as_posix(), 0, 0, 0, None, val)

    def symbol(val: str) -> SymbolNode:
        return SymbolNode(token('', val))

    def string(value: str) -> StringNode:
        return StringNode(token(val=value), escape=False)

    def id_node(value: str) -> IdNode:
        return IdNode(token(val=value))

    def number(value: int) -> NumberNode:
        return NumberNode(token(val=str(value)))

    def nodeify(value: TYPE_mixed_list) -> BaseNode:
        if isinstance(value, str):
            return string(value)
        if isinstance(value, Path):
            return string(value.as_posix())
        elif isinstance(value, bool):
            return BooleanNode(token(val=value))
        elif isinstance(value, int):
            return number(value)
        elif isinstance(value, list):
            return array(value)
        elif isinstance(value, BaseNode):
            return value
        raise RuntimeError('invalid type of value: {} ({})'.format(type(value).__name__, str(value)))

    def indexed(node: BaseNode, index: int) -> IndexNode:
        return IndexNode(node, symbol('['), nodeify(index), symbol(']'))

    def array(elements: TYPE_mixed_list) -> ArrayNode:
        args = ArgumentNode(token())
        if not isinstance(elements, list):
            elements = [args]
        args.arguments += [nodeify(x) for x in elements if x is not None]
        return ArrayNode(symbol('['), args, symbol(']'))

    def function(name: str, args: T.Optional[TYPE_mixed_list]=None, kwargs: T.Optional[TYPE_mixed_kwargs]=None) -> FunctionNode:
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        args_n = ArgumentNode(token())
        if not isinstance(args, list):
            assert isinstance(args, (str, int, bool, Path, BaseNode))
            args = [args]
        args_n.arguments = [nodeify(x) for x in args if x is not None]
        args_n.kwargs = {id_node(k): nodeify(v) for k, v in kwargs.items() if v is not None}
        func_n = FunctionNode(id_node(name), symbol('('), args_n, symbol(')'))
        return func_n

    def method(obj: BaseNode, name: str, args: T.Optional[TYPE_mixed_list]=None, kwargs: T.Optional[TYPE_mixed_kwargs]=None) -> MethodNode:
        args = [] if args is None else args
        kwargs = {} if kwargs is None else kwargs
        args_n = ArgumentNode(token())
        if not isinstance(args, list):
            assert isinstance(args, (str, int, bool, Path, BaseNode))
            args = [args]
        args_n.arguments = [nodeify(x) for x in args if x is not None]
        args_n.kwargs = {id_node(k): nodeify(v) for k, v in kwargs.items() if v is not None}
        return MethodNode(obj, symbol('.'), id_node(name), symbol('('), args_n, symbol(')'))

    def assign(var_name: str, value: BaseNode) -> AssignmentNode:
        return AssignmentNode(id_node(var_name), symbol('='), value)
    root_cb = CodeBlockNode(token())
    root_cb.lines += [function('project', [self.project_name] + self.languages, {'version': self.project_version} if self.project_version else None)]
    processing: T.List[str] = []
    processed: T.Dict[str, T.Dict[str, T.Optional[str]]] = {}
    name_map: T.Dict[str, str] = {}

    def extract_tgt(tgt: T.Union[ConverterTarget, ConverterCustomTarget, CustomTargetReference]) -> IdNode:
        tgt_name = None
        if isinstance(tgt, (ConverterTarget, ConverterCustomTarget)):
            tgt_name = tgt.name
        elif isinstance(tgt, CustomTargetReference):
            tgt_name = tgt.ctgt.name
        assert tgt_name is not None and tgt_name in processed
        res_var = processed[tgt_name]['tgt']
        return id_node(res_var) if res_var else None

    def detect_cycle(tgt: T.Union[ConverterTarget, ConverterCustomTarget]) -> None:
        if tgt.name in processing:
            raise CMakeException('Cycle in CMake inputs/dependencies detected')
        processing.append(tgt.name)

    def resolve_ctgt_ref(ref: CustomTargetReference) -> T.Union[IdNode, IndexNode]:
        tgt_var = extract_tgt(ref)
        if len(ref.ctgt.outputs) == 1:
            return tgt_var
        else:
            return indexed(tgt_var, ref.index)

    def process_target(tgt: ConverterTarget) -> None:
        detect_cycle(tgt)
        link_with: T.List[IdNode] = []
        objec_libs: T.List[IdNode] = []
        sources: T.List[Path] = []
        generated: T.List[T.Union[IdNode, IndexNode]] = []
        generated_filenames: T.List[str] = []
        custom_targets: T.List[ConverterCustomTarget] = []
        dependencies: T.List[IdNode] = []
        for i in tgt.link_with:
            assert isinstance(i, ConverterTarget)
            if i.name not in processed:
                process_target(i)
            link_with += [extract_tgt(i)]
        for i in tgt.object_libs:
            assert isinstance(i, ConverterTarget)
            if i.name not in processed:
                process_target(i)
            objec_libs += [extract_tgt(i)]
        for i in tgt.depends:
            if not isinstance(i, ConverterCustomTarget):
                continue
            if i.name not in processed:
                process_custom_target(i)
            dependencies += [extract_tgt(i)]
        sources += tgt.sources
        sources += tgt.generated
        for ctgt_ref in tgt.generated_ctgt:
            ctgt = ctgt_ref.ctgt
            if ctgt.name not in processed:
                process_custom_target(ctgt)
            generated += [resolve_ctgt_ref(ctgt_ref)]
            generated_filenames += [ctgt_ref.filename()]
            if ctgt not in custom_targets:
                custom_targets += [ctgt]
        for ctgt in custom_targets:
            for j in ctgt.outputs:
                if not is_header(j) or j in generated_filenames:
                    continue
                generated += [resolve_ctgt_ref(ctgt.get_ref(Path(j)))]
                generated_filenames += [j]
        tgt_func = tgt.meson_func()
        if not tgt_func:
            raise CMakeException(f'Unknown target type "{tgt.type}"')
        inc_var = f'{tgt.name}_inc'
        dir_var = f'{tgt.name}_dir'
        sys_var = f'{tgt.name}_sys'
        src_var = f'{tgt.name}_src'
        dep_var = f'{tgt.name}_dep'
        tgt_var = tgt.name
        install_tgt = options.get_install(tgt.cmake_name, tgt.install)
        tgt_kwargs: TYPE_mixed_kwargs = {'build_by_default': install_tgt, 'link_args': options.get_link_args(tgt.cmake_name, tgt.link_flags + tgt.link_libraries), 'link_with': link_with, 'include_directories': id_node(inc_var), 'install': install_tgt, 'override_options': options.get_override_options(tgt.cmake_name, tgt.override_options), 'objects': [method(x, 'extract_all_objects') for x in objec_libs]}
        if install_tgt and tgt.install_dir:
            tgt_kwargs['install_dir'] = tgt.install_dir
        for key, val in tgt.compile_opts.items():
            tgt_kwargs[f'{key}_args'] = options.get_compile_args(tgt.cmake_name, key, val)
        if tgt_func == 'executable':
            tgt_kwargs['pie'] = tgt.pie
        elif tgt_func == 'static_library':
            tgt_kwargs['pic'] = tgt.pie
        dep_kwargs: TYPE_mixed_kwargs = {'link_args': tgt.link_flags + tgt.link_libraries, 'link_with': id_node(tgt_var), 'compile_args': tgt.public_compile_opts, 'include_directories': id_node(inc_var)}
        if dependencies:
            generated += dependencies
        dir_node = assign(dir_var, function('include_directories', tgt.includes))
        sys_node = assign(sys_var, function('include_directories', tgt.sys_includes, {'is_system': True}))
        inc_node = assign(inc_var, array([id_node(dir_var), id_node(sys_var)]))
        node_list = [dir_node, sys_node, inc_node]
        if tgt_func == 'header_only':
            del dep_kwargs['link_with']
            dep_node = assign(dep_var, function('declare_dependency', kwargs=dep_kwargs))
            node_list += [dep_node]
            src_var = None
            tgt_var = None
        else:
            src_node = assign(src_var, function('files', sources))
            tgt_node = assign(tgt_var, function(tgt_func, [tgt_var, id_node(src_var), *generated], tgt_kwargs))
            node_list += [src_node, tgt_node]
            if tgt_func in {'static_library', 'shared_library'}:
                dep_node = assign(dep_var, function('declare_dependency', kwargs=dep_kwargs))
                node_list += [dep_node]
            elif tgt_func == 'shared_module':
                del dep_kwargs['link_with']
                dep_node = assign(dep_var, function('declare_dependency', kwargs=dep_kwargs))
                node_list += [dep_node]
            else:
                dep_var = None
        root_cb.lines += node_list
        processed[tgt.name] = {'inc': inc_var, 'src': src_var, 'dep': dep_var, 'tgt': tgt_var, 'func': tgt_func}
        name_map[tgt.cmake_name] = tgt.name

    def process_custom_target(tgt: ConverterCustomTarget) -> None:
        detect_cycle(tgt)
        tgt_var = tgt.name

        def resolve_source(x: T.Union[str, ConverterTarget, ConverterCustomTarget, CustomTargetReference]) -> T.Union[str, IdNode, IndexNode]:
            if isinstance(x, ConverterTarget):
                if x.name not in processed:
                    process_target(x)
                return extract_tgt(x)
            if isinstance(x, ConverterCustomTarget):
                if x.name not in processed:
                    process_custom_target(x)
                return extract_tgt(x)
            elif isinstance(x, CustomTargetReference):
                if x.ctgt.name not in processed:
                    process_custom_target(x.ctgt)
                return resolve_ctgt_ref(x)
            else:
                return x
        command: T.List[T.Union[str, IdNode, IndexNode]] = []
        command += mesonlib.get_meson_command()
        command += ['--internal', 'cmake_run_ctgt']
        command += ['-o', '@OUTPUT@']
        if tgt.original_outputs:
            command += ['-O'] + [x.as_posix() for x in tgt.original_outputs]
        command += ['-d', tgt.working_dir.as_posix()]
        for cmd in tgt.command:
            command += [resolve_source(x) for x in cmd] + [';;;']
        tgt_kwargs: TYPE_mixed_kwargs = {'input': [resolve_source(x) for x in tgt.inputs], 'output': tgt.outputs, 'command': command, 'depends': [resolve_source(x) for x in tgt.depends]}
        root_cb.lines += [assign(tgt_var, function('custom_target', [tgt.name], tgt_kwargs))]
        processed[tgt.name] = {'inc': None, 'src': None, 'dep': None, 'tgt': tgt_var, 'func': 'custom_target'}
        name_map[tgt.cmake_name] = tgt.name
    for ctgt in self.custom_targets:
        if ctgt.name not in processed:
            process_custom_target(ctgt)
    for tgt in self.targets:
        if tgt.name not in processed:
            process_target(tgt)
    self.generated_targets = processed
    self.internal_name_map = name_map
    return root_cb