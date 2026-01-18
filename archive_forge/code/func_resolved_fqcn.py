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
@property
def resolved_fqcn(self):
    if not self.resolved:
        return
    if not self._resolved_fqcn:
        final_plugin = self.redirect_list[-1]
        if AnsibleCollectionRef.is_valid_fqcr(final_plugin) and final_plugin.startswith('ansible.legacy.'):
            final_plugin = final_plugin.split('ansible.legacy.')[-1]
        if self.plugin_resolved_collection and (not AnsibleCollectionRef.is_valid_fqcr(final_plugin)):
            final_plugin = self.plugin_resolved_collection + '.' + final_plugin
        self._resolved_fqcn = final_plugin
    return self._resolved_fqcn