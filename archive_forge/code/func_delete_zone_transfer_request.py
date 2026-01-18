from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def delete_zone_transfer_request(self, request, ignore_missing=True):
    """Delete a ZoneTransfer Request

        :param request: The value can be the ID of a zone transfer request
            or a :class:`~openstack.dns.v2.zone_transfer.ZoneTransferRequest`
            instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the zone does not exist.
            When set to ``True``, no exception will be set when attempting to
            delete a nonexistent zone.

        :returns: None
        """
    return self._delete(_zone_transfer.ZoneTransferRequest, request, ignore_missing=ignore_missing)