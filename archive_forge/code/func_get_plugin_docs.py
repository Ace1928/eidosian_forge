from __future__ import (absolute_import, division, print_function)
from collections.abc import MutableMapping, MutableSet, MutableSequence
from pathlib import Path
from ansible import constants as C
from ansible.release import __version__ as ansible_version
from ansible.errors import AnsibleError, AnsibleParserError, AnsiblePluginNotFound
from ansible.module_utils.six import string_types
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.plugin_docs import read_docstring
from ansible.parsing.yaml.loader import AnsibleLoader
from ansible.utils.display import Display
def get_plugin_docs(plugin, plugin_type, loader, fragment_loader, verbose):
    docs = []
    filename, collection_name = find_plugin_docfile(plugin, plugin_type, loader)
    try:
        docs = get_docstring(filename, fragment_loader, verbose=verbose, collection_name=collection_name, plugin_type=plugin_type)
    except Exception as e:
        raise AnsibleParserError('%s did not contain a DOCUMENTATION attribute (%s)' % (plugin, filename), orig_exc=e)
    if not docs[0]:
        for newfile in _find_adjacent(filename, plugin, C.DOC_EXTENSIONS):
            try:
                docs = get_docstring(newfile, fragment_loader, verbose=verbose, collection_name=collection_name, plugin_type=plugin_type)
                filename = newfile
                if docs[0] is not None:
                    break
            except Exception as e:
                raise AnsibleParserError('Adjacent file %s did not contain a DOCUMENTATION attribute (%s)' % (plugin, filename), orig_exc=e)
    if docs[0] is None:
        raise AnsibleParserError('No documentation available for %s (%s)' % (plugin, filename))
    else:
        docs[0]['filename'] = filename
        docs[0]['collection'] = collection_name
    return docs