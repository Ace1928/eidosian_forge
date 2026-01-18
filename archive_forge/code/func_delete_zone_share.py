from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def delete_zone_share(self, zone, zone_share, ignore_missing=True):
    """Delete a zone share

        :param zone: The zone ID or a
            :class:`~openstack.dns.v2.zone.Zone` instance
        :param zone_share: The zone_share can be either the ID of the zone
            share or a :class:`~openstack.dns.v2.zone_share.ZoneShare` instance
            that the zone share belongs to.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the zone share does not exist.
            When set to ``True``, no exception will be set when attempting to
            delete a nonexistent zone share.

        :returns: ``None``
        """
    zone_obj = self._get_resource(_zone.Zone, zone)
    self._delete(_zone_share.ZoneShare, zone_share, ignore_missing=ignore_missing, zone_id=zone_obj.id)