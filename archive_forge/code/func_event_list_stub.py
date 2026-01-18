from unittest import mock
import testtools
from heatclient.common import event_utils
from heatclient.v1 import events as hc_ev
from heatclient.v1 import resources as hc_res
def event_list_stub(stack_id, argfoo):
    return [events[stack_id]]