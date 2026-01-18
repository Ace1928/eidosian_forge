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
def get_contained_plugins(self, collection, plugin_path, name):
    plugins = []
    full_name = '.'.join(['ansible_collections', collection, 'plugins', self.type, name])
    try:
        if plugin_path not in self._module_cache:
            self._module_cache[plugin_path] = self._load_module_source(full_name, plugin_path)
        module = self._module_cache[plugin_path]
        obj = getattr(module, self.class_name)
    except Exception as e:
        raise KeyError('Failed to load %s for %s: %s' % (plugin_path, collection, to_native(e)))
    plugin_impl = obj()
    if plugin_impl is None:
        raise KeyError('Could not find %s.%s' % (collection, name))
    try:
        method_map = getattr(plugin_impl, self.method_map_name)
        plugin_map = method_map().items()
    except Exception as e:
        display.warning("Ignoring %s plugins in '%s' as it seems to be invalid: %r" % (self.type, to_text(plugin_path), e))
        return plugins
    for func_name, func in plugin_map:
        fq_name = '.'.join((collection, func_name))
        full = '.'.join((full_name, func_name))
        plugin = self._plugin_wrapper_type(func)
        if plugin in plugins:
            continue
        self._update_object(plugin, full, plugin_path, resolved=fq_name)
        plugins.append(plugin)
    return plugins