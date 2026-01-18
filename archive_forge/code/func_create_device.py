import datetime
import logging
import os
import shutil
import time
import numpy
import pygloo
import ray
from ray._private import ray_constants
from ray.util.collective.collective_group import gloo_util
from ray.util.collective.collective_group.base_collective_group import BaseGroup
from ray.util.collective.const import get_store_name
from ray.util.collective.types import (
def create_device(self, device_type):
    if device_type == 'tcp':
        attr = pygloo.transport.tcp.attr(self._process_ip_address)
        self._device = pygloo.transport.tcp.CreateDevice(attr)
    elif device_type == 'uv':
        raise NotImplementedError('No implementation for uv.')