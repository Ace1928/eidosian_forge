import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def serialize_input_data(input_data):
    return {'input_data': _serialize(input_data)}