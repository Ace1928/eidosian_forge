from __future__ import (absolute_import, division, print_function)
from ansible.cli import CLI
import pkgutil
import os
import os.path
import re
import textwrap
import traceback
import ansible.plugins.loader as plugin_loader
from pathlib import Path
from ansible import constants as C
from ansible import context
from ansible.cli.arguments import option_helpers as opt_help
from ansible.collections.list import list_collection_dirs
from ansible.errors import AnsibleError, AnsibleOptionsError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.collections import is_sequence
from ansible.module_utils.common.json import json_dump
from ansible.module_utils.common.yaml import yaml_dump
from ansible.module_utils.compat import importlib
from ansible.module_utils.six import string_types
from ansible.parsing.plugin_docs import read_docstub
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.yaml.dumper import AnsibleDumper
from ansible.plugins.list import list_plugins
from ansible.plugins.loader import action_loader, fragment_loader
from ansible.utils.collection_loader import AnsibleCollectionConfig, AnsibleCollectionRef
from ansible.utils.collection_loader._collection_finder import _get_collection_name_from_path
from ansible.utils.display import Display
from ansible.utils.plugin_docs import get_plugin_docs, get_docstring, get_versioned_doclink
@staticmethod
def get_plugin_metadata(plugin_type, plugin_name):
    loader = getattr(plugin_loader, '%s_loader' % plugin_type)
    result = loader.find_plugin_with_context(plugin_name, mod_type='.py', ignore_deprecated=True, check_aliases=True)
    if not result.resolved:
        raise AnsibleError('unable to load {0} plugin named {1} '.format(plugin_type, plugin_name))
    filename = result.plugin_resolved_path
    collection_name = result.plugin_resolved_collection
    try:
        doc, __, __, __ = get_docstring(filename, fragment_loader, verbose=context.CLIARGS['verbosity'] > 0, collection_name=collection_name, plugin_type=plugin_type)
    except Exception:
        display.vvv(traceback.format_exc())
        raise AnsibleError('%s %s at %s has a documentation formatting error or is missing documentation.' % (plugin_type, plugin_name, filename))
    if doc is None:
        return None
    return dict(name=plugin_name, namespace=DocCLI.namespace_from_plugin_filepath(filename, plugin_name, loader.package_path), description=doc.get('short_description', 'UNKNOWN'), version_added=doc.get('version_added', 'UNKNOWN'))