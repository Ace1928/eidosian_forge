from __future__ import annotations
import collections, functools, importlib
import typing as T
from .base import ExternalDependency, DependencyException, DependencyMethods, NotFoundDependency
from ..mesonlib import listify, MachineChoice, PerMachine
from .. import mlog
def find_external_dependency(name: str, env: 'Environment', kwargs: T.Dict[str, object], candidates: T.Optional[T.List['DependencyGenerator']]=None) -> T.Union['ExternalDependency', NotFoundDependency]:
    assert name
    required = kwargs.get('required', True)
    if not isinstance(required, bool):
        raise DependencyException('Keyword "required" must be a boolean.')
    if not isinstance(kwargs.get('method', ''), str):
        raise DependencyException('Keyword "method" must be a string.')
    lname = name.lower()
    if lname not in _packages_accept_language and 'language' in kwargs:
        raise DependencyException(f'{name} dependency does not accept "language" keyword argument')
    if not isinstance(kwargs.get('version', ''), (str, list)):
        raise DependencyException('Keyword "Version" must be string or list.')
    display_name = display_name_map.get(lname, lname)
    for_machine = MachineChoice.BUILD if kwargs.get('native', False) else MachineChoice.HOST
    type_text = PerMachine('Build-time', 'Run-time')[for_machine] + ' dependency'
    if candidates is None:
        candidates = _build_external_dependency_list(name, env, for_machine, kwargs)
    pkg_exc: T.List[DependencyException] = []
    pkgdep: T.List[ExternalDependency] = []
    details = ''
    for c in candidates:
        try:
            d = c()
            d._check_version()
            pkgdep.append(d)
        except DependencyException as e:
            assert isinstance(c, functools.partial), 'for mypy'
            bettermsg = f'Dependency lookup for {name} with method {c.func.log_tried()!r} failed: {e}'
            mlog.debug(bettermsg)
            e.args = (bettermsg,)
            pkg_exc.append(e)
        else:
            pkg_exc.append(None)
            details = d.log_details()
            if details:
                details = '(' + details + ') '
            if 'language' in kwargs:
                details += 'for ' + d.language + ' '
            if d.found():
                info: mlog.TV_LoggableList = []
                if d.version:
                    info.append(mlog.normal_cyan(d.version))
                log_info = d.log_info()
                if log_info:
                    info.append('(' + log_info + ')')
                mlog.log(type_text, mlog.bold(display_name), details + 'found:', mlog.green('YES'), *info)
                return d
    tried_methods = [d.log_tried() for d in pkgdep if d.log_tried()]
    if tried_methods:
        tried = mlog.format_list(tried_methods)
    else:
        tried = ''
    mlog.log(type_text, mlog.bold(display_name), details + 'found:', mlog.red('NO'), f'(tried {tried})' if tried else '')
    if required:
        if pkg_exc and pkg_exc[0]:
            raise pkg_exc[0]
        raise DependencyException(f'Dependency "{name}" not found' + (f', tried {tried}' if tried else ''))
    return NotFoundDependency(name, env)