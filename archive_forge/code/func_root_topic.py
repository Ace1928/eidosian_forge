from __future__ import annotations
import logging
from copy import copy
import zmq
@root_topic.setter
def root_topic(self, value: str):
    self.setRootTopic(value)