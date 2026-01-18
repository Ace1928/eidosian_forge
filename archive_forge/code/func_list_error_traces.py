import collections
import datetime
import time
from urllib import parse as parser
from oslo_config import cfg
from oslo_serialization import jsonutils
from osprofiler import _utils as utils
from osprofiler.drivers import base
from osprofiler import exc
def list_error_traces(self):
    """Please use Jaeger Tracing UI for this task."""
    return []