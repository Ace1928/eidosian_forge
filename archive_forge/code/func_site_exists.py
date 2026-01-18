from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def site_exists(self):
    """
        Check if the site exists in Cisco Catalyst Center.

        Parameters:
          - self (object): An instance of the class containing the method.
        Returns:
          - tuple: A tuple containing a boolean indicating whether the site exists and
                   a dictionary containing information about the existing site.
                   The returned tuple includes two elements:
                   - site_exists (bool): Indicates whether the site exists.
                   - dict: Contains information about the existing site. If the
                           site doesn't exist, this dictionary is empty.
        Description:
            Checks the existence of a site in Cisco Catalyst Center by querying the
          'get_site' function in the 'sites' family. It utilizes the
          'site_name' parameter from the 'want' attribute to identify the site.
        """
    site_exists = False
    current_site = {}
    response = None
    try:
        response = self.dnac._exec(family='sites', function='get_site', params={'name': self.want.get('site_name')})
    except Exception as e:
        self.log("The provided site name '{0}' is either invalid or not present in the Cisco Catalyst Center.".format(self.want.get('site_name')), 'WARNING')
    if response:
        response = response.get('response')
        self.log("Received API response from 'get_site': {0}".format(str(response)), 'DEBUG')
        current_site = self.get_current_site(response)
        site_exists = True
        self.log("Site '{0}' exists in Cisco Catalyst Center".format(self.want.get('site_name')), 'INFO')
    return (site_exists, current_site)