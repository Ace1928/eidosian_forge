from __future__ import (absolute_import, division, print_function)
import os
from ansible import context
from ansible import constants as C
from ansible.collections.list import list_collections
from ansible.errors import AnsibleError
from ansible.module_utils.common.text.converters import to_native, to_bytes
from ansible.plugins import loader
from ansible.utils.display import Display
from ansible.utils.collection_loader._collection_finder import _get_collection_path
def list_collection_plugins(ptype, collections, search_paths=None):
    plugins = {}
    try:
        ploader = getattr(loader, '{0}_loader'.format(ptype))
    except AttributeError:
        raise AnsibleError('Cannot list plugins, incorrect plugin type supplied: {0}'.format(ptype))
    for collection in collections.keys():
        if collection == 'ansible.builtin':
            dirs = [d.path for d in ploader._get_paths_with_context() if d.internal]
        elif collection == 'ansible.legacy':
            dirs = [d.path for d in ploader._get_paths_with_context() if not d.internal]
            if context.CLIARGS.get('module_path', None):
                dirs.extend(context.CLIARGS['module_path'])
        else:
            b_ptype = to_bytes(C.COLLECTION_PTYPE_COMPAT.get(ptype, ptype))
            dirs = [to_native(os.path.join(collections[collection], b'plugins', b_ptype))]
        plugins.update(_list_plugins_from_paths(ptype, dirs, collection))
    if ptype in ('module',):
        for plugin in plugins.keys():
            plugins[plugin] = (plugins[plugin], None)
    else:
        for plugin in list(plugins.keys()):
            pobj = None
            try:
                pobj = ploader.get(plugin, class_only=True)
            except Exception as e:
                display.vvv("The '{0}' {1} plugin could not be loaded from '{2}': {3}".format(plugin, ptype, plugins[plugin], to_native(e)))
            plugins[plugin] = (plugins[plugin], pobj)
    return plugins