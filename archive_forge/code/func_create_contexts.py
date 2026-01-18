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
def create_contexts(self):
    for key, cls in self.contexts_cls.items():
        if issubclass(cls, OSKenApp):
            context = self._instantiate(None, cls)
        else:
            context = cls()
        LOG.info('creating context %s', key)
        assert key not in self.contexts
        self.contexts[key] = context
    return self.contexts