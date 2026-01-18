from __future__ import absolute_import, division, print_function
import copy
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_site_id(self, site_name):
    """
        Get the site id from the site name.
        Use check_return_status() to check for failure

        Parameters:
            site_name (str) - Site name

        Returns:
            str or None - The Site Id if found, or None if not found or error
        """
    try:
        response = self.dnac._exec(family='sites', function='get_site', params={'name': site_name})
        self.log("Received API response from 'get_site': {0}".format(response), 'DEBUG')
        if not response:
            self.log('Failed to retrieve the site ID for the site name: {0}'.format(site_name), 'ERROR')
            return None
        _id = response.get('response')[0].get('id')
        self.log("Site ID for site name '{0}': {1}".format(site_name, _id), 'DEBUG')
    except Exception as msg:
        self.log('Exception occurred while retrieving site_id from the site_name: {0}'.format(msg), 'CRITICAL')
        return None
    return _id