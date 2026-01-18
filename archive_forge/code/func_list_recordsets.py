from openstack.cloud import _utils
from openstack.dns.v2._proxy import Proxy
from openstack import exceptions
from openstack import resource
def list_recordsets(self, zone):
    """List all available recordsets.

        :param zone: Name, ID or :class:`openstack.dns.v2.zone.Zone` instance
            of the zone managing the recordset.

        :returns: A list of recordsets.

        """
    if isinstance(zone, resource.Resource):
        zone_obj = zone
    else:
        zone_obj = self.get_zone(zone)
    if zone_obj is None:
        raise exceptions.SDKException('Zone %s not found.' % zone)
    return list(self.dns.recordsets(zone_obj))