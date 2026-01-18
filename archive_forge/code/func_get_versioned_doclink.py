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
def get_versioned_doclink(path):
    """
    returns a versioned documentation link for the current Ansible major.minor version; used to generate
    in-product warning/error links to the configured DOCSITE_ROOT_URL
    (eg, https://docs.ansible.com/ansible/2.8/somepath/doc.html)

    :param path: relative path to a document under docs/docsite/rst;
    :return: absolute URL to the specified doc for the current version of Ansible
    """
    path = to_native(path)
    try:
        base_url = C.config.get_config_value('DOCSITE_ROOT_URL')
        if not base_url.endswith('/'):
            base_url += '/'
        if path.startswith('/'):
            path = path[1:]
        split_ver = ansible_version.split('.')
        if len(split_ver) < 3:
            raise RuntimeError('invalid version ({0})'.format(ansible_version))
        doc_version = '{0}.{1}'.format(split_ver[0], split_ver[1])
        if split_ver[2].startswith('0'):
            if any((pre in split_ver[2] for pre in ['a', 'b'])) or (len(split_ver) > 3 and 'dev' in split_ver[3]):
                doc_version = 'devel'
        return '{0}{1}/{2}'.format(base_url, doc_version, path)
    except Exception as ex:
        return '(unable to create versioned doc link for path {0}: {1})'.format(path, to_native(ex))