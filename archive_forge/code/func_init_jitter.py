import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def init_jitter(existing_input_data):
    nconflicts = max(0, len(predecessors) - len(existing_input_data) - 1)
    return min(nconflicts, 1000) * 0.01