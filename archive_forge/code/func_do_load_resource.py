from ansible.module_utils.network.aos.aos import (check_aos_version, get_aos_session, find_collection_item,
from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.network.plugins.module_utils.version import LooseVersion
from ansible.module_utils._text import to_native
def do_load_resource(module, collection, name):
    """
    Create a new object (collection.item) by loading a datastructure directly
    """
    try:
        item = find_collection_item(collection, name, '')
    except Exception:
        module.fail_json(msg="An error occurred while running 'find_collection_item'")
    if item.exists:
        module.exit_json(changed=False, name=item.name, id=item.id, value=item.value)
    if not module.check_mode:
        try:
            item.datum = module.params['content']
            item.write()
        except Exception as e:
            module.fail_json(msg='Unable to write item content : %r' % to_native(e))
    module.exit_json(changed=True, name=item.name, id=item.id, value=item.value)