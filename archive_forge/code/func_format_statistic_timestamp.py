import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_statistic_timestamp(statistics, columns, name):
    time_index = 0
    if statistics:
        time_index = columns.index(name)
    time_list = list()
    for timestamp in statistics:
        time_list.append(str(timestamp[time_index]))
    return '\n'.join(time_list)