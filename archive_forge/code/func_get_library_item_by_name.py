from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import env_fallback, missing_required_lib
from ansible.module_utils._text import to_native
def get_library_item_by_name(self, name):
    """
        Returns the identifier of the library item with the given name.

        Args:
            name (str): The name of item to look for

        Returns:
            str: The item ID or None if the item is not found
        """
    find_spec = Item.FindSpec(name=name)
    item_ids = self.api_client.content.library.Item.find(find_spec)
    item_id = item_ids[0] if item_ids else None
    return item_id