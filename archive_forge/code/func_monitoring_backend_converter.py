from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from typing import Dict
from apitools.base.protorpclite import messages
def monitoring_backend_converter(msg: messages.Message) -> MessageMap:
    return {'prometheus': msg.ConfigManagementPolicyControllerMonitoring.BackendsValueListEntryValuesEnum.PROMETHEUS, 'cloudmonitoring': msg.ConfigManagementPolicyControllerMonitoring.BackendsValueListEntryValuesEnum.CLOUD_MONITORING, 'cloud_monitoring': msg.ConfigManagementPolicyControllerMonitoring.BackendsValueListEntryValuesEnum.CLOUD_MONITORING}