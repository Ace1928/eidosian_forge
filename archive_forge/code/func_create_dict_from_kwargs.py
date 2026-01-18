import sys
import time
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_serialization import jsonutils as json
from oslo_utils import timeutils
from oslo_utils import uuidutils
from saharaclient.api import base
def create_dict_from_kwargs(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}