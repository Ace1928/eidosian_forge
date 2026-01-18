import logging
import os
import time
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, Optional
import ray
from ray._private.resource_spec import NODE_ID_PREFIX
def summary_string(self):
    summary = '{} CPUs, {} GPUs'.format(self.cpu + self.extra_cpu, self.gpu + self.extra_gpu)
    if self.memory or self.extra_memory:
        summary += ', {} GiB heap'.format(round((self.memory + self.extra_memory) / 1024 ** 3, 2))
    if self.object_store_memory or self.extra_object_store_memory:
        summary += ', {} GiB objects'.format(round((self.object_store_memory + self.extra_object_store_memory) / 1024 ** 3, 2))
    custom_summary = ', '.join(['{} {}'.format(self.get_res_total(res), res) for res in self.custom_resources if not res.startswith(NODE_ID_PREFIX)])
    if custom_summary:
        summary += ' ({})'.format(custom_summary)
    return summary