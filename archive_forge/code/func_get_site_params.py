from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.cisco.dnac.plugins.module_utils.dnac import (
def get_site_params(self, params):
    """
        Store the site-related parameters.

        Parameters:
          self (object): An instance of a class used for interacting with Cisco Catalyst Center.
          - params (dict): Dictionary containing site-related parameters.
        Returns:
          - dict: Dictionary containing the stored site-related parameters.
                  The returned dictionary includes the following keys:
                  - 'type' (str): The type of the site.
                  - 'site' (dict): Dictionary containing site-related info.
        Description:
            This method takes a dictionary 'params' containing site-related
          information and stores the relevant parameters based on the site
          type. If the site type is 'floor', it ensures that the 'rfModel'
          parameter is stored in uppercase.
        """
    typeinfo = params.get('type')
    site_info = {}
    if typeinfo == 'area':
        area_details = params.get('site').get('area')
        site_info['area'] = {'name': area_details.get('name'), 'parentName': area_details.get('parent_name')}
    elif typeinfo == 'building':
        building_details = params.get('site').get('building')
        site_info['building'] = {'name': building_details.get('name'), 'address': building_details.get('address'), 'parentName': building_details.get('parent_name'), 'latitude': building_details.get('latitude'), 'longitude': building_details.get('longitude'), 'country': building_details.get('country')}
    else:
        floor_details = params.get('site').get('floor')
        site_info['floor'] = {'name': floor_details.get('name'), 'parentName': floor_details.get('parent_name'), 'length': floor_details.get('length'), 'width': floor_details.get('width'), 'height': floor_details.get('height'), 'floorNumber': floor_details.get('floor_number', '')}
        try:
            site_info['floor']['rfModel'] = floor_details.get('rf_model')
        except Exception as e:
            self.log("The attribute 'rf_model' is missing in floor '{0}'.".format(floor_details.get('name')), 'WARNING')
    site_params = dict(type=typeinfo, site=site_info)
    self.log('Site parameters: {0}'.format(str(site_params)), 'DEBUG')
    return site_params