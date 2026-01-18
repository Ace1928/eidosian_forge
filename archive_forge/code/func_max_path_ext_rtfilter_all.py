import logging
import numbers
from os_ken.lib import ip
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp import rtconf
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import validate
@property
def max_path_ext_rtfilter_all(self):
    return self._settings[MAX_PATH_EXT_RTFILTER_ALL]