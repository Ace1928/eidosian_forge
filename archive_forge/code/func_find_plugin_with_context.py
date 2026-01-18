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
def find_plugin_with_context(self, name, mod_type='', ignore_deprecated=False, check_aliases=False, collection_list=None):
    """ Find a plugin named name, returning contextual info about the load, recursively resolving redirection """
    plugin_load_context = PluginLoadContext()
    plugin_load_context.original_name = name
    while True:
        result = self._resolve_plugin_step(name, mod_type, ignore_deprecated, check_aliases, collection_list, plugin_load_context=plugin_load_context)
        if result.pending_redirect:
            if result.pending_redirect in result.redirect_list:
                raise AnsiblePluginCircularRedirect('plugin redirect loop resolving {0} (path: {1})'.format(result.original_name, result.redirect_list))
            name = result.pending_redirect
            result.pending_redirect = None
            plugin_load_context = result
        else:
            break
    if plugin_load_context.error_list:
        display.warning('errors were encountered during the plugin load for {0}:\n{1}'.format(name, plugin_load_context.error_list))
    return plugin_load_context