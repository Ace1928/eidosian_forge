from __future__ import absolute_import, division, print_function
import time
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_site_type(self, site_name_hierarchy=None):
    """
        Fetches the type of site

        Parameters:
          - self: The instance of the class containing the 'config' attribute
                  to be validated.
        Returns:
          The method returns an instance of the class with updated attributes:
          - site_type: A string indicating the type of the
                       site (area/building/floor).
        Example:
          Post creation of the validated input, we this method gets the
          type of the site.
        """
    try:
        response = self.dnac_apply['exec'](family='sites', function='get_site', params={'name': site_name_hierarchy})
    except Exception:
        self.log("Exception occurred as                 site '{0}' was not found".format(self.want.get('site_name')), 'CRITICAL')
        self.module.fail_json(msg='Site not found', response=[])
    if response:
        self.log("Received site details                for '{0}': {1}".format(site_name_hierarchy, str(response)), 'DEBUG')
        site = response.get('response')
        site_additional_info = site[0].get('additionalInfo')
        for item in site_additional_info:
            if item['nameSpace'] == 'Location':
                site_type = item.get('attributes').get('type')
                self.log("Site type for site name '{1}' : {0}".format(site_type, site_name_hierarchy), 'INFO')
    return site_type