import datetime
import json
import logging
import time
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
def logd(**kwargs):
    log(log_level=logging.DEBUG, **kwargs)