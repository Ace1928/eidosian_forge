import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def register_view(self, view):
    """register_view will create the needed structure
        in order to be able to sent all data to Prometheus
        """
    v_name = get_view_name(self.options.namespace, view)
    if v_name not in self.registered_views:
        desc = {'name': v_name, 'documentation': view.description, 'labels': list(map(sanitize, view.columns)), 'units': view.measure.unit}
        self.registered_views[v_name] = desc