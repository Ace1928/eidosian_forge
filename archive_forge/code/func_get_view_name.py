import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def get_view_name(namespace, view):
    """create the name for the view"""
    name = ''
    if namespace != '':
        name = namespace + '_'
    return sanitize(name + view.name)