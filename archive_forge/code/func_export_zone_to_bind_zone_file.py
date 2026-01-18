import datetime
from typing import Any, Dict, List, Type, Union, Iterator, Optional
from libcloud import __version__
from libcloud.dns.types import RecordType
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
def export_zone_to_bind_zone_file(self, zone, file_path):
    """
        Export Zone object to the BIND compatible format and write result to a
        file.

        :param zone: Zone to export.
        :type  zone: :class:`Zone`

        :param file_path: File path where the output will be saved.
        :type  file_path: ``str``
        """
    result = self.export_zone_to_bind_format(zone=zone)
    with open(file_path, 'w') as fp:
        fp.write(result)