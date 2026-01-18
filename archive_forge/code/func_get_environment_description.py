import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def get_environment_description(header):
    import sysconfig
    import site
    result = [header, '\n\n']

    def report(s, *args, **kwargs):
        result.append(s.format(*args, **kwargs))

    def report_paths(get_paths, label=None):
        prefix = f'    {label or get_paths}: '
        expr = None
        if not callable(get_paths):
            expr = get_paths
            get_paths = lambda: util.evaluate(expr)
        try:
            paths = get_paths()
        except AttributeError:
            report('{0}<missing>\n', prefix)
            return
        except Exception:
            swallow_exception('Error evaluating {0}', repr(expr) if expr else util.srcnameof(get_paths), level='info')
            return
        if not isinstance(paths, (list, tuple)):
            paths = [paths]
        for p in sorted(paths):
            report('{0}{1}', prefix, p)
            if p is not None:
                rp = os.path.realpath(p)
                if p != rp:
                    report('({0})', rp)
            report('\n')
            prefix = ' ' * len(prefix)
    report('System paths:\n')
    report_paths('sys.executable')
    report_paths('sys.prefix')
    report_paths('sys.base_prefix')
    report_paths('sys.real_prefix')
    report_paths('site.getsitepackages()')
    report_paths('site.getusersitepackages()')
    site_packages = [p for p in sys.path if os.path.exists(p) and os.path.basename(p) == 'site-packages']
    report_paths(lambda: site_packages, 'sys.path (site-packages)')
    for name in sysconfig.get_path_names():
        expr = 'sysconfig.get_path({0!r})'.format(name)
        report_paths(expr)
    report_paths('os.__file__')
    report_paths('threading.__file__')
    report_paths('debugpy.__file__')
    report('\n')
    importlib_metadata = None
    try:
        import importlib_metadata
    except ImportError:
        try:
            from importlib import metadata as importlib_metadata
        except ImportError:
            pass
    if importlib_metadata is None:
        report('Cannot enumerate installed packages - missing importlib_metadata.')
    else:
        report('Installed packages:\n')
        try:
            for pkg in importlib_metadata.distributions():
                report('    {0}=={1}\n', pkg.name, pkg.version)
        except Exception:
            swallow_exception('Error while enumerating installed packages.', level='info')
    return ''.join(result).rstrip('\n')