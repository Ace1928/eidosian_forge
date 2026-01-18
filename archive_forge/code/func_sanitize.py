import re
from prometheus_client import start_http_server
from prometheus_client.core import (
from opencensus.common.transports import sync
from opencensus.stats import aggregation_data as aggregation_data_module
from opencensus.stats import base_exporter
import logging
def sanitize(key):
    """sanitize the given metric name or label according to Prometheus rule.
    Replace all characters other than [A-Za-z0-9_] with '_'.
    """
    return _NON_LETTERS_NOR_DIGITS_RE.sub('_', key)