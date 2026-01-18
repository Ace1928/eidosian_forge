import logging
from functools import partial
from collections import defaultdict
from os.path import splitext
from .diagrams_base import BaseGraph
def set_previous_transition(self, src, dst):
    src_name = self._get_global_name(src.split(self.machine.state_cls.separator))
    dst_name = self._get_global_name(dst.split(self.machine.state_cls.separator))
    super(NestedGraph, self).set_previous_transition(src_name, dst_name)