import inspect
import itertools
import logging
import sys
import os
import gc
from os_ken import cfg
from os_ken import utils
from os_ken.controller.handler import register_instance, get_dependent_services
from os_ken.controller.controller import Datapath
from os_ken.controller import event
from os_ken.controller.event import EventRequestBase, EventReplyBase
from os_ken.lib import hub
from os_ken.ofproto import ofproto_protocol
def set_main_thread(self, thread):
    """
        Set self.main_thread so that stop() can terminate it.

        Only AppManager.instantiate_apps should call this function.
        """
    self.main_thread = thread