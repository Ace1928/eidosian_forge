from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_zone_transfer_accept(self, accept):
    """Get a ZoneTransfer Accept info

        :param request: The value can be the ID of a transfer accept
            or a :class:`~openstack.dns.v2.zone_transfer.ZoneTransferAccept`
            instance.
        :returns: Zone transfer request instance.
        :rtype: :class:`~openstack.dns.v2.zone_transfer.ZoneTransferAccept`
        """
    return self._get(_zone_transfer.ZoneTransferAccept, accept)