import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def new_collector(options):
    """new_collector should be used
    to create instance of Collector class in order to
    prevent the usage of constructor directly
    """
    return Collector(options=options)