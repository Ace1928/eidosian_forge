import collections
import threading
import kombu
from kombu import exceptions as kombu_exceptions
from taskflow.engines.worker_based import dispatcher
from taskflow import logging
@property
def connection_details(self):
    """Details about the connection (read-only)."""
    driver_version = self._conn.transport.driver_version()
    if driver_version and driver_version.lower() == 'n/a':
        driver_version = None
    if self._conn.transport_options:
        transport_options = self._conn.transport_options.copy()
    else:
        transport_options = {}
    transport = _TransportDetails(options=transport_options, driver_type=self._conn.transport.driver_type, driver_name=self._conn.transport.driver_name, driver_version=driver_version)
    return _ConnectionDetails(uri=self._conn.as_uri(include_password=False), transport=transport)