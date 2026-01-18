import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_measure_value(measurements):
    return '\n'.join(['{:12.3f}'.format(m[1]) for m in measurements])