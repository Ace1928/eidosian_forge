from __future__ import (absolute_import, division, print_function)
import glob
import os
import os.path
import pkgutil
import sys
import warnings
from collections import defaultdict, namedtuple
from traceback import format_exc
import ansible.module_utils.compat.typing as t
from .filter import AnsibleJinja2Filter
from .test import AnsibleJinja2Test
from ansible import __version__ as ansible_version
from ansible import constants as C
from ansible.errors import AnsibleError, AnsiblePluginCircularRedirect, AnsiblePluginRemovedError, AnsibleCollectionUnsupportedVersionError
from ansible.module_utils.common.text.converters import to_bytes, to_text, to_native
from ansible.module_utils.compat.importlib import import_module
from ansible.module_utils.six import string_types
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.plugins import get_plugin_class, MODULE_CACHE, PATH_CACHE, PLUGIN_PATH_CACHE
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _AnsibleCollectionFinder, _get_collection_metadata
from ansible.utils.display import Display
from ansible.utils.plugin_docs import add_fragments
from ansible.utils.unsafe_proxy import _is_unsafe
import importlib.util
def get_shell_plugin(shell_type=None, executable=None):
    if not shell_type:
        shell_type = 'sh'
        if executable:
            if isinstance(executable, string_types):
                shell_filename = os.path.basename(executable)
                try:
                    shell = shell_loader.get(shell_filename)
                except Exception:
                    shell = None
                if shell is None:
                    for shell in shell_loader.all():
                        if shell_filename in shell.COMPATIBLE_SHELLS:
                            shell_type = shell.SHELL_FAMILY
                            break
        else:
            raise AnsibleError('Either a shell type or a shell executable must be provided ')
    shell = shell_loader.get(shell_type)
    if not shell:
        raise AnsibleError('Could not find the shell plugin required (%s).' % shell_type)
    if executable:
        setattr(shell, 'executable', executable)
    return shell