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
def register_observer(self, ev_cls, name, states=None):
    states = states or set()
    ev_cls_observers = self.observers.setdefault(ev_cls, {})
    ev_cls_observers.setdefault(name, set()).update(states)