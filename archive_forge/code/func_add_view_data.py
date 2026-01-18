import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def add_view_data(self, view_data):
    """Add view data object to be sent to server"""
    self.register_view(view_data.view)
    v_name = get_view_name(self.options.namespace, view_data.view)
    self.view_name_to_data_map[v_name] = view_data