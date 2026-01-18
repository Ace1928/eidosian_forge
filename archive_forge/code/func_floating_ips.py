from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def floating_ips(self, **query):
    """Retrieve a generator of recordsets

        :param dict query: Optional query parameters to be sent to limit the
            resources being returned.

            * `name`: Recordset Name field.
            * `type`: Type field.
            * `status`: Status of the recordset.
            * `ttl`: TTL field filter.
            * `description`: Recordset description field filter.

        :returns: A generator of floatingips
            (:class:`~openstack.dns.v2.floating_ip.FloatingIP`) instances
        """
    return self._list(_fip.FloatingIP, **query)