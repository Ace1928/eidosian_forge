from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
@staticmethod
def merge_dict_values(norm_current_values, norm_updated_values):
    """ Create an merged update object for manageiq group filters.

            The input dict contain the tag values per category.
            If the new values contain the category, all tags for that category are replaced
            If the new values do not contain the category, the existing tags are kept

        Returns:
            the nested array with the merged values, used in the update post body
        """
    if norm_current_values and (not norm_updated_values):
        return norm_current_values
    if not norm_current_values and norm_updated_values:
        return norm_updated_values
    res = norm_current_values.copy()
    res.update(norm_updated_values)
    return res