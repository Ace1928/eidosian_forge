from __future__ import absolute_import, division, print_function
from collections import namedtuple
def set_site_bd_subnet(self, subnet, fail_module=True):
    """
        Get site bridge domain subnet item that matches the ip of a subnet.
        :param subnet: Subnet (ip) to match. -> Str
        :param fail_module: When match is not found fail the ansible module. -> Bool
        :return: Site bd subnet item. -> Item(Int, Dict) | None
        """
    self.validate_schema_objects_present(['site_bd'])
    kv_list = [KVPair('ip', subnet)]
    match, existing = self.get_object_from_list(self.schema_objects['site_bd'].details.get('subnets'), kv_list)
    if not match and fail_module:
        msg = "Provided subnet '{0}' not matching existing site bd subnet(s): {1}".format(subnet, ', '.join(existing))
        self.mso.fail_json(msg=msg)
    self.schema_objects['site_bd_subnet'] = match