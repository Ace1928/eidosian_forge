from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def site_requires_update(self):
    """
        Check if the site requires updates.
        Parameters:
            self (object): An instance of a class used for interacting with Cisco Catalyst Center.
        Returns:
            bool: True if the site requires updates, False otherwise.
        Description:
            This method compares the site parameters of the current site
            ('current_site') and the requested site parameters ('requested_site')
            stored in the 'want' attribute. It checks for differences in
            specified parameters, such as the site type and site details.
        """
    type = self.have['current_site']['type']
    updated_site = self.have['current_site']['site'][type]
    requested_site = self.want['site_params']['site'][type]
    self.log('Current Site type: {0}'.format(str(updated_site)), 'INFO')
    self.log('Requested Site type: {0}'.format(str(requested_site)), 'INFO')
    if type == 'building':
        return not self.is_building_updated(updated_site, requested_site)
    elif type == 'floor':
        return not self.is_floor_updated(updated_site, requested_site)
    return not self.is_area_updated(updated_site, requested_site)