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
def get_role_man_text(self, role, role_json):
    """Generate text for the supplied role suitable for display.

        This is similar to get_man_text(), but roles are different enough that we have
        a separate method for formatting their display.

        :param role: The role name.
        :param role_json: The JSON for the given role as returned from _create_role_doc().

        :returns: A array of text suitable for displaying to screen.
        """
    text = []
    opt_indent = '        '
    pad = display.columns * 0.2
    limit = max(display.columns - int(pad), 70)
    text.append('> %s    (%s)\n' % (role.upper(), role_json.get('path')))
    for entry_point in role_json['entry_points']:
        doc = role_json['entry_points'][entry_point]
        if doc.get('short_description'):
            text.append('ENTRY POINT: %s - %s\n' % (entry_point, doc.get('short_description')))
        else:
            text.append('ENTRY POINT: %s\n' % entry_point)
        if doc.get('description'):
            if isinstance(doc['description'], list):
                desc = ' '.join(doc['description'])
            else:
                desc = doc['description']
            text.append('%s\n' % textwrap.fill(DocCLI.tty_ify(desc), limit, initial_indent=opt_indent, subsequent_indent=opt_indent))
        if doc.get('options'):
            text.append('OPTIONS (= is mandatory):\n')
            DocCLI.add_fields(text, doc.pop('options'), limit, opt_indent)
            text.append('')
        if doc.get('attributes'):
            text.append('ATTRIBUTES:\n')
            text.append(DocCLI._indent_lines(DocCLI._dump_yaml(doc.pop('attributes')), opt_indent))
            text.append('')
        for k in ('author',):
            if k not in doc:
                continue
            if isinstance(doc[k], string_types):
                text.append('%s: %s' % (k.upper(), textwrap.fill(DocCLI.tty_ify(doc[k]), limit - (len(k) + 2), subsequent_indent=opt_indent)))
            elif isinstance(doc[k], (list, tuple)):
                text.append('%s: %s' % (k.upper(), ', '.join(doc[k])))
            else:
                text.append(DocCLI._indent_lines(DocCLI._dump_yaml({k.upper(): doc[k]}), ''))
            text.append('')
    return text