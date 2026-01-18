from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def find_zone_share(self, zone, zone_share_id, ignore_missing=True):
    """Find a single zone share

        :param zone: The value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance.
        :param zone_share_id: The zone share ID
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised
            when the zone share does not exist.
            When set to ``True``,  None will be returned when attempting to
            find a nonexistent zone share.

        :returns: :class:`~openstack.dns.v2.zone_share.ZoneShare`
        """
    zone_obj = self._get_resource(_zone.Zone, zone)
    return self._find(_zone_share.ZoneShare, zone_share_id, ignore_missing=ignore_missing, zone_id=zone_obj.id)