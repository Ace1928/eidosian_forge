from openstack.dns.v2 import floating_ip as _fip
from openstack.dns.v2 import recordset as _rs
from openstack.dns.v2 import zone as _zone
from openstack.dns.v2 import zone_export as _zone_export
from openstack.dns.v2 import zone_import as _zone_import
from openstack.dns.v2 import zone_share as _zone_share
from openstack.dns.v2 import zone_transfer as _zone_transfer
from openstack import proxy
def get_zone_export_text(self, zone_export):
    """Get a zone export record as text

        :param zone: The value can be the ID of a zone import
            or a :class:`~openstack.dns.v2.zone_export.ZoneExport` instance.
        :returns: ZoneExport instance.
        :rtype: :class:`~openstack.dns.v2.zone_export.ZoneExport`
        """
    return self._get(_zone_export.ZoneExport, zone_export, base_path='/zones/tasks/export/%(id)s/export')