from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_site_name(self, site):
    """
        Get and Return the site name.
        Parameters:
          - self (object): An instance of a class used for interacting with Cisco Catalyst Center.
          - site (dict): A dictionary containing information about the site.
        Returns:
          - str: The constructed site name.
        Description:
            This method takes a dictionary 'site' containing information about
          the site and constructs the site name by combining the parent name
          and site name.
        """
    site_type = site.get('type')
    parent_name = site.get('site').get(site_type).get('parent_name')
    name = site.get('site').get(site_type).get('name')
    site_name = '/'.join([parent_name, name])
    self.log('Site name: {0}'.format(site_name), 'INFO')
    return site_name