import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_metric_dimensions(metrics):
    metric_string_list = list()
    for metric in metrics:
        metric_dimensions = metric['dimensions']
        for k, v in metric_dimensions.items():
            if isinstance(v, numbers.Number):
                d_str = k + ': ' + str(v)
            else:
                d_str = k + ': ' + v
            metric_string_list.append(d_str)
    return '\n'.join(metric_string_list)