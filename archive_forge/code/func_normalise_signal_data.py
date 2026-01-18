from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine.resources import signal_responder
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
def normalise_signal_data(self, signal_data, latest_metadata):
    signal_num = len(latest_metadata) + 1
    reason = 'Signal %s received' % signal_num
    metadata = signal_data.copy() if signal_data else {}
    metadata.setdefault(self.REASON, reason)
    metadata.setdefault(self.DATA, None)
    metadata.setdefault(self.UNIQUE_ID, signal_num)
    metadata.setdefault(self.STATUS, self.STATUS_SUCCESS)
    return metadata