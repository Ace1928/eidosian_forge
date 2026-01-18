from libcloud.utils.py3 import ET, tostring
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.common.abiquo import AbiquoResponse, AbiquoConnection, get_href
from libcloud.compute.types import Provider, LibcloudError
def ex_populate_cache(self):
    """
        Populate the cache.

        For each connection, it is good to store some objects that will be
        useful for further requests, such as the 'user' and the 'enterprise'
        objects.

        Executes the 'login' resource after setting the connection parameters
        and, if the execution is successful, it sets the 'user' object into
        cache. After that, it also requests for the 'enterprise' and
        'locations' data.

        List of locations should remain the same for a single libcloud
        connection. However, this method is public and you are able to
        refresh the list of locations any time.
        """
    user_headers = {'Accept': self.USER_MIME_TYPE}
    user = self.connection.request('/login', headers=user_headers).object
    self.connection.cache['user'] = user
    e_ent = get_href(self.connection.cache['user'], 'enterprise')
    ent_headers = {'Accept': self.ENT_MIME_TYPE}
    ent = self.connection.request(e_ent, headers=ent_headers).object
    self.connection.cache['enterprise'] = ent
    vdcs_headers = {'Accept': self.VDCS_MIME_TYPE}
    uri_vdcs = '/cloud/virtualdatacenters'
    e_vdcs = self.connection.request(uri_vdcs, headers=vdcs_headers).object
    params = {'idEnterprise': self._get_enterprise_id()}
    dcs_headers = {'Accept': self.DCS_MIME_TYPE}
    e_dcs = self.connection.request('/admin/datacenters', headers=dcs_headers, params=params).object
    dc_dict = {}
    for dc in e_dcs.findall('datacenter'):
        key = get_href(dc, 'self')
        dc_dict[key] = dc
    self.connection.cache['locations'] = {}
    for e_vdc in e_vdcs.findall('virtualDatacenter'):
        loc = get_href(e_vdc, 'location')
        if loc is not None:
            self.connection.cache['locations'][loc] = get_href(e_vdc, 'edit')