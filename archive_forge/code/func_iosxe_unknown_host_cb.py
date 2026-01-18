from .default import DefaultDeviceHandler
from ncclient.operations.third_party.iosxe.rpc import SaveConfig
from ncclient.xml_ import BASE_NS_1_0
import logging
def iosxe_unknown_host_cb(host, fingerprint):
    return True