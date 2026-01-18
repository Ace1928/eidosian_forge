from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def recordsets(self, zone=None, **query):
    """Retrieve a generator of recordsets

        :param zone: The optional value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance. If it is not
            given all recordsets for all zones of the tenant would be
            retrieved
        :param dict query: Optional query parameters to be sent to limit the
            resources being returned.

            * `name`: Recordset Name field.
            * `type`: Type field.
            * `status`: Status of the recordset.
            * `ttl`: TTL field filter.
            * `description`: Recordset description field filter.

        :returns: A generator of zone
            (:class:`~openstack.dns.v2.recordset.Recordset`) instances
        """
    base_path = None
    if not zone:
        base_path = '/recordsets'
    else:
        zone = self._get_resource(_zone.Zone, zone)
        query.update({'zone_id': zone.id})
    return self._list(_rs.Recordset, base_path=base_path, **query)