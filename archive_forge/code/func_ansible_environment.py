from __future__ import annotations
import json
import os
import shutil
import typing as t
from .constants import (
from .io import (
from .util import (
from .util_common import (
from .config import (
from .data import (
from .python_requirements import (
from .host_configs import (
from .thread import (
def ansible_environment(args: CommonConfig, color: bool=True, ansible_config: t.Optional[str]=None) -> dict[str, str]:
    """Return a dictionary of environment variables to use when running Ansible commands."""
    env = common_environment()
    path = env['PATH']
    ansible_bin_path = get_ansible_bin_path(args)
    if not path.startswith(ansible_bin_path + os.path.pathsep):
        path = ansible_bin_path + os.path.pathsep + path
    if not ansible_config:
        ansible_config = args.get_ansible_config()
    if not args.explain and (not os.path.exists(ansible_config)):
        raise ApplicationError('Configuration not found: %s' % ansible_config)
    ansible = dict(ANSIBLE_PYTHON_MODULE_RLIMIT_NOFILE=str(SOFT_RLIMIT_NOFILE), ANSIBLE_FORCE_COLOR='%s' % 'true' if args.color and color else 'false', ANSIBLE_FORCE_HANDLERS='true', ANSIBLE_HOST_PATTERN_MISMATCH='error', ANSIBLE_INVENTORY='/dev/null', ANSIBLE_DEPRECATION_WARNINGS='false', ANSIBLE_HOST_KEY_CHECKING='false', ANSIBLE_RETRY_FILES_ENABLED='false', ANSIBLE_CONFIG=ansible_config, ANSIBLE_LIBRARY='/dev/null', ANSIBLE_DEVEL_WARNING='false', PYTHONPATH=get_ansible_python_path(args), PAGER='/bin/cat', PATH=path, ANSIBLE_WORKER_SHUTDOWN_POLL_COUNT='100', ANSIBLE_WORKER_SHUTDOWN_POLL_DELAY='0.1')
    if isinstance(args, IntegrationConfig) and args.coverage:
        ansible.update(ANSIBLE_CONNECTION_PATH=os.path.join(get_injector_path(), 'ansible-connection'))
    if isinstance(args, PosixIntegrationConfig):
        ansible.update(ANSIBLE_PYTHON_INTERPRETER='/set/ansible_python_interpreter/in/inventory')
    env.update(ansible)
    if args.debug:
        env.update(ANSIBLE_DEBUG='true', ANSIBLE_LOG_PATH=os.path.join(ResultType.LOGS.name, 'debug.log'))
    if data_context().content.collection:
        env.update(ANSIBLE_COLLECTIONS_PATH=data_context().content.collection.root)
    if data_context().content.is_ansible:
        env.update(configure_plugin_paths(args))
    return env