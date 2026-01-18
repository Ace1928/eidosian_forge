from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_site_details(self):
    """
        Check whether the site exists or not, along with side id

        Parameters:
          - self: The instance of the class containing the 'config'
                  attribute to be validated.
        Returns:
          The method returns an instance of the class with updated attributes:
          - site_exits: A boolean value indicating the existence of the site.
          - site_id: The Id of the site i.e. required to claim device to site.
        Example:
          Post creation of the validated input, we this method gets the
          site_id and checks whether the site exists or not
        """
    site_exists = False
    site_id = None
    response = None
    try:
        response = self.dnac_apply['exec'](family='sites', function='get_site', params={'name': self.want.get('site_name')})
    except Exception:
        self.log("Exception occurred as site                 '{0}' was not found".format(self.want.get('site_name')), 'CRITICAL')
        self.module.fail_json(msg='Site not found', response=[])
    if response:
        self.log("Received site details                 for '{0}': {1}".format(self.want.get('site_name'), str(response)), 'DEBUG')
        site = response.get('response')
        if len(site) == 1:
            site_id = site[0].get('id')
            site_exists = True
            self.log('Site Name: {1}, Site ID: {0}'.format(site_id, self.want.get('site_name')), 'INFO')
    return (site_exists, site_id)