import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
def batch_action(self, *args, **kwargs):
    return action(self, *args, **kwargs)