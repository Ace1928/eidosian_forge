import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_metric_name(metrics):
    metric_string_list = list()
    for metric in metrics:
        metric_name = metric['name']
        metric_dimensions = metric['dimensions']
        metric_string_list.append(metric_name)
        rng = len(metric_dimensions)
        for i in range(rng):
            if i == rng - 1:
                break
            metric_string_list.append(' ')
    return '\n'.join(metric_string_list)