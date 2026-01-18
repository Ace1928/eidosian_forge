import datetime
import numbers
import time
from keystoneauth1 import exceptions as k_exc
from osc_lib import exceptions as osc_exc
from monascaclient.common import utils
from oslo_serialization import jsonutils
def format_value_meta(measurements):
    measure_string_list = list()
    for measure in measurements:
        if len(measure) < 3:
            measure_string = ''
        else:
            meta_string_list = []
            for k, v in measure[2].items():
                if isinstance(v, numbers.Number):
                    m_str = k + ': ' + str(v)
                else:
                    m_str = k + ': ' + v
                meta_string_list.append(m_str)
            measure_string = ','.join(meta_string_list)
        measure_string_list.append(measure_string)
    return '\n'.join(measure_string_list)