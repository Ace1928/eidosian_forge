from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def delete_recordset(self, recordset, zone=None, ignore_missing=True):
    """Delete a zone

        :param recordset: The value can be the ID of a recordset
            or a :class:`~openstack.dns.v2.recordset.Recordset`
            instance.
        :param zone: The value can be the ID of a zone
            or a :class:`~openstack.dns.v2.zone.Zone` instance.
        :param bool ignore_missing: When set to ``False``
            :class:`~openstack.exceptions.ResourceNotFound` will be raised when
            the zone does not exist. When set to ``True``, no exception will
            be set when attempting to delete a nonexistent zone.

        :returns: Recordset instance been deleted
        :rtype: :class:`~openstack.dns.v2.recordset.Recordset`
        """
    if zone:
        zone = self._get_resource(_zone.Zone, zone)
        recordset = self._get(_rs.Recordset, recordset, zone_id=zone.id)
    return self._delete(_rs.Recordset, recordset, ignore_missing=ignore_missing)