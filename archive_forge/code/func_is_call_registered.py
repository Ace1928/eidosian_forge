import logging
import traceback
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import API_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
def is_call_registered(call_name):
    return call_name in _CALL_REGISTRY