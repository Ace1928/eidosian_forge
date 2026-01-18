import datetime
from typing import Any, Dict, List, Type, Union, Iterator, Optional
from libcloud import __version__
from libcloud.dns.types import RecordType
from libcloud.common.base import BaseDriver, Connection, ConnectionUserAndKey
def export_to_bind_zone_file(self, file_path):
    self.driver.export_zone_to_bind_zone_file(zone=self, file_path=file_path)