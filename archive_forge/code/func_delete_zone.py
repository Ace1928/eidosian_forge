from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def delete_zone(self, zone, ignore_missing=True, delete_shares=False):
    """Delete a zone

        :param zone: The value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the zone does not exist.
            When set to ``True``, no exception will be set when attempting to
            delete a nonexistent zone.
        :param bool delete_shares: When True, delete the zone shares along with
                                   the zone.

        :returns: Zone been deleted
        :rtype: :class:`~openstack.dns.v2.zone.Zone`
        """
    return self._delete(_zone.Zone, zone, ignore_missing=ignore_missing, delete_shares=delete_shares)