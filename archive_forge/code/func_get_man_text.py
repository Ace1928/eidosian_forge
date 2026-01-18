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
def get_man_text(doc, collection_name='', plugin_type=''):
    doc = dict(doc)
    DocCLI.IGNORE = DocCLI.IGNORE + (context.CLIARGS['type'],)
    opt_indent = '        '
    text = []
    pad = display.columns * 0.2
    limit = max(display.columns - int(pad), 70)
    plugin_name = doc.get(context.CLIARGS['type'], doc.get('name')) or doc.get('plugin_type') or plugin_type
    if collection_name:
        plugin_name = '%s.%s' % (collection_name, plugin_name)
    text.append('> %s    (%s)\n' % (plugin_name.upper(), doc.pop('filename')))
    if isinstance(doc['description'], list):
        desc = ' '.join(doc.pop('description'))
    else:
        desc = doc.pop('description')
    text.append('%s\n' % textwrap.fill(DocCLI.tty_ify(desc), limit, initial_indent=opt_indent, subsequent_indent=opt_indent))
    if 'version_added' in doc:
        version_added = doc.pop('version_added')
        version_added_collection = doc.pop('version_added_collection', None)
        text.append('ADDED IN: %s\n' % DocCLI._format_version_added(version_added, version_added_collection))
    if doc.get('deprecated', False):
        text.append('DEPRECATED: \n')
        if isinstance(doc['deprecated'], dict):
            if 'removed_at_date' in doc['deprecated']:
                text.append('\tReason: %(why)s\n\tWill be removed in a release after %(removed_at_date)s\n\tAlternatives: %(alternative)s' % doc.pop('deprecated'))
            else:
                if 'version' in doc['deprecated'] and 'removed_in' not in doc['deprecated']:
                    doc['deprecated']['removed_in'] = doc['deprecated']['version']
                text.append('\tReason: %(why)s\n\tWill be removed in: Ansible %(removed_in)s\n\tAlternatives: %(alternative)s' % doc.pop('deprecated'))
        else:
            text.append('%s' % doc.pop('deprecated'))
        text.append('\n')
    if doc.pop('has_action', False):
        text.append('  * note: %s\n' % 'This module has a corresponding action plugin.')
    if doc.get('options', False):
        text.append('OPTIONS (= is mandatory):\n')
        DocCLI.add_fields(text, doc.pop('options'), limit, opt_indent)
        text.append('')
    if doc.get('attributes', False):
        text.append('ATTRIBUTES:\n')
        text.append(DocCLI._indent_lines(DocCLI._dump_yaml(doc.pop('attributes')), opt_indent))
        text.append('')
    if doc.get('notes', False):
        text.append('NOTES:')
        for note in doc['notes']:
            text.append(textwrap.fill(DocCLI.tty_ify(note), limit - 6, initial_indent=opt_indent[:-2] + '* ', subsequent_indent=opt_indent))
        text.append('')
        text.append('')
        del doc['notes']
    if doc.get('seealso', False):
        text.append('SEE ALSO:')
        for item in doc['seealso']:
            if 'module' in item:
                text.append(textwrap.fill(DocCLI.tty_ify('Module %s' % item['module']), limit - 6, initial_indent=opt_indent[:-2] + '* ', subsequent_indent=opt_indent))
                description = item.get('description')
                if description is None and item['module'].startswith('ansible.builtin.'):
                    description = 'The official documentation on the %s module.' % item['module']
                if description is not None:
                    text.append(textwrap.fill(DocCLI.tty_ify(description), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
                if item['module'].startswith('ansible.builtin.'):
                    relative_url = 'collections/%s_module.html' % item['module'].replace('.', '/', 2)
                    text.append(textwrap.fill(DocCLI.tty_ify(get_versioned_doclink(relative_url)), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent))
            elif 'plugin' in item and 'plugin_type' in item:
                plugin_suffix = ' plugin' if item['plugin_type'] not in ('module', 'role') else ''
                text.append(textwrap.fill(DocCLI.tty_ify('%s%s %s' % (item['plugin_type'].title(), plugin_suffix, item['plugin'])), limit - 6, initial_indent=opt_indent[:-2] + '* ', subsequent_indent=opt_indent))
                description = item.get('description')
                if description is None and item['plugin'].startswith('ansible.builtin.'):
                    description = 'The official documentation on the %s %s%s.' % (item['plugin'], item['plugin_type'], plugin_suffix)
                if description is not None:
                    text.append(textwrap.fill(DocCLI.tty_ify(description), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
                if item['plugin'].startswith('ansible.builtin.'):
                    relative_url = 'collections/%s_%s.html' % (item['plugin'].replace('.', '/', 2), item['plugin_type'])
                    text.append(textwrap.fill(DocCLI.tty_ify(get_versioned_doclink(relative_url)), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent))
            elif 'name' in item and 'link' in item and ('description' in item):
                text.append(textwrap.fill(DocCLI.tty_ify(item['name']), limit - 6, initial_indent=opt_indent[:-2] + '* ', subsequent_indent=opt_indent))
                text.append(textwrap.fill(DocCLI.tty_ify(item['description']), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
                text.append(textwrap.fill(DocCLI.tty_ify(item['link']), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
            elif 'ref' in item and 'description' in item:
                text.append(textwrap.fill(DocCLI.tty_ify('Ansible documentation [%s]' % item['ref']), limit - 6, initial_indent=opt_indent[:-2] + '* ', subsequent_indent=opt_indent))
                text.append(textwrap.fill(DocCLI.tty_ify(item['description']), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
                text.append(textwrap.fill(DocCLI.tty_ify(get_versioned_doclink('/#stq=%s&stp=1' % item['ref'])), limit - 6, initial_indent=opt_indent + '   ', subsequent_indent=opt_indent + '   '))
        text.append('')
        text.append('')
        del doc['seealso']
    if doc.get('requirements', False):
        req = ', '.join(doc.pop('requirements'))
        text.append('REQUIREMENTS:%s\n' % textwrap.fill(DocCLI.tty_ify(req), limit - 16, initial_indent='  ', subsequent_indent=opt_indent))
    for k in sorted(doc):
        if k in DocCLI.IGNORE or not doc[k]:
            continue
        if isinstance(doc[k], string_types):
            text.append('%s: %s' % (k.upper(), textwrap.fill(DocCLI.tty_ify(doc[k]), limit - (len(k) + 2), subsequent_indent=opt_indent)))
        elif isinstance(doc[k], (list, tuple)):
            text.append('%s: %s' % (k.upper(), ', '.join(doc[k])))
        else:
            text.append(DocCLI._indent_lines(DocCLI._dump_yaml({k.upper(): doc[k]}), ''))
        del doc[k]
        text.append('')
    if doc.get('plainexamples', False):
        text.append('EXAMPLES:')
        text.append('')
        if isinstance(doc['plainexamples'], string_types):
            text.append(doc.pop('plainexamples').strip())
        else:
            try:
                text.append(yaml_dump(doc.pop('plainexamples'), indent=2, default_flow_style=False))
            except Exception as e:
                raise AnsibleParserError('Unable to parse examples section', orig_exc=e)
        text.append('')
        text.append('')
    if doc.get('returndocs', False):
        text.append('RETURN VALUES:')
        DocCLI.add_fields(text, doc.pop('returndocs'), limit, opt_indent, return_values=True)
    return '\n'.join(text)