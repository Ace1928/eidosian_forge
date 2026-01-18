import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_statistic_value(statistics, columns, stat_type):
    stat_index = 0
    if statistics:
        stat_index = columns.index(stat_type)
    value_list = list()
    for stat in statistics:
        value_str = '{:12.3f}'.format(stat[stat_index])
        value_list.append(value_str)
    return '\n'.join(value_list)