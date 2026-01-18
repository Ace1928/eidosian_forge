import collections
import time
from openstack.cloud import meta
from openstack import exceptions
def stop_check_action(a):
    stop_status = ('%s_FAILED' % action, '%s_COMPLETE' % action)
    return a in stop_status