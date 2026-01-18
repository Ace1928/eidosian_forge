from __future__ import absolute_import, division, print_function
import base64
import json
import os
from copy import deepcopy
from ansible.module_utils.urls import fetch_url
from ansible.module_utils._text import to_bytes, to_native
from ansible.module_utils.basic import env_fallback
from ansible.module_utils.connection import Connection
def get_diff_children(self, aci_class, proposed_obj=None, existing_obj=None):
    """
        This method is used to retrieve the updated child configs by comparing the proposed children configs
        against the objects existing children configs.
        :param aci_class: Type str.
                          This is the root dictionary key for the MO's configuration body, or the ACI class of the MO.
        :return: The list of updated child config dictionaries. None is returned if there are no changes to the child
                 configurations.
        """
    if proposed_obj is None:
        proposed_children = self.proposed[aci_class].get('children')
    else:
        proposed_children = proposed_obj
    if proposed_children:
        child_updates = []
        if existing_obj is None:
            existing_children = self.existing[0][aci_class].get('children', [])
        else:
            existing_children = existing_obj
        for child in proposed_children:
            child_class, proposed_child, existing_child = self.get_nested_config(child, existing_children)
            proposed_child_children, existing_child_children = self.get_nested_children(child, existing_children)
            if existing_child is None:
                child_update = child
            else:
                child_update = self.get_diff_child(child_class, proposed_child, existing_child)
                if proposed_child_children:
                    child_update_children = self.get_diff_children(aci_class, proposed_child_children, existing_child_children)
                    if child_update_children:
                        child_update = child
            if child_update:
                child_updates.append(child_update)
    else:
        return None
    return child_updates