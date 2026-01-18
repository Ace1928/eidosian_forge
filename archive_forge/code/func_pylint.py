from __future__ import annotations
import collections.abc as c
import itertools
import json
import os
import datetime
import configparser
import typing as t
from . import (
from ...constants import (
from ...io import (
from ...test import (
from ...target import (
from ...util import (
from ...util_common import (
from ...ansible_util import (
from ...config import (
from ...data import (
from ...host_configs import (
@staticmethod
def pylint(args: SanityConfig, context: str, paths: list[str], plugin_dir: str, plugin_names: list[str], python: PythonConfig, collection_detail: CollectionDetail, min_python_version_db_path: str) -> list[dict[str, str]]:
    """Run pylint using the config specified by the context on the specified paths."""
    rcfile = os.path.join(SANITY_ROOT, 'pylint', 'config', context.split('/')[0] + '.cfg')
    if not os.path.exists(rcfile):
        if data_context().content.collection:
            rcfile = os.path.join(SANITY_ROOT, 'pylint', 'config', 'collection.cfg')
        else:
            rcfile = os.path.join(SANITY_ROOT, 'pylint', 'config', 'default.cfg')
    parser = configparser.ConfigParser()
    parser.read(rcfile)
    if parser.has_section('ansible-test'):
        config = dict(parser.items('ansible-test'))
    else:
        config = {}
    disable_plugins = set((i.strip() for i in config.get('disable-plugins', '').split(',') if i))
    load_plugins = set(plugin_names + ['pylint.extensions.mccabe']) - disable_plugins
    cmd = [python.path, '-m', 'pylint', '--jobs', '0', '--reports', 'n', '--max-line-length', '160', '--max-complexity', '20', '--rcfile', rcfile, '--output-format', 'json', '--load-plugins', ','.join(sorted(load_plugins)), '--min-python-version-db', min_python_version_db_path] + paths
    if data_context().content.collection:
        cmd.extend(['--collection-name', data_context().content.collection.full_name])
        if collection_detail and collection_detail.version:
            cmd.extend(['--collection-version', collection_detail.version])
    append_python_path = [plugin_dir]
    if data_context().content.collection:
        append_python_path.append(data_context().content.collection.root)
    env = ansible_environment(args)
    env['PYTHONPATH'] += os.path.pathsep + os.path.pathsep.join(append_python_path)
    env.update(dict((('ANSIBLE_TEST_%s_PATH' % k.upper(), os.path.abspath(v) + os.path.sep) for k, v in data_context().content.plugin_paths.items())))
    pylint_home = os.path.join(ResultType.TMP.path, 'pylint')
    make_dirs(pylint_home)
    env.update(PYLINTHOME=pylint_home)
    if paths:
        display.info('Checking %d file(s) in context "%s" with config: %s' % (len(paths), context, rcfile), verbosity=1)
        try:
            stdout, stderr = run_command(args, cmd, env=env, capture=True)
            status = 0
        except SubprocessError as ex:
            stdout = ex.stdout
            stderr = ex.stderr
            status = ex.status
        if stderr or status >= 32:
            raise SubprocessError(cmd=cmd, status=status, stderr=stderr, stdout=stdout)
    else:
        stdout = None
    if not args.explain and stdout:
        messages = json.loads(stdout)
    else:
        messages = []
    return messages