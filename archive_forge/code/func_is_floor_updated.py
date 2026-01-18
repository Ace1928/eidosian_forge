from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def is_floor_updated(self, updated_site, requested_site):
    """
        Check if the floor details in a site have been updated.

        Args:
            - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
            - updated_site (dict): The site details after the update.
            - requested_site (dict): The site details as requested for the update.
        Return:
            bool: True if the floor details have been updated, False otherwise.
        Description:
            This method compares the floor details of the updated site with the requested site.
            It checks if the name, rf_model, length, width, and height are equal, indicating
            that the floor details have been updated. Returns True if the details match, and False otherwise.
        """
    keys_to_compare = ['length', 'width', 'height']
    if updated_site['name'] != requested_site['name'] or updated_site['rf_model'] != requested_site['rfModel']:
        return False
    for key in keys_to_compare:
        if not self.compare_float_values(updated_site[key], requested_site[key]):
            return False
    return True