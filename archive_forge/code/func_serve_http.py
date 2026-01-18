import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def serve_http(self):
    """serve_http serves the Prometheus endpoint."""
    address = str(self.options.address)
    kwargs = {'addr': address} if address else {}
    start_http_server(port=self.options.port, **kwargs)