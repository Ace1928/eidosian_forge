import ast
import tenacity
from oslo_log import log as logging
from heat.common import exception
from heat.objects import sync_point as sync_point_object
def deserialize_input_data(db_input_data):
    db_input_data = db_input_data.get('input_data')
    if not db_input_data:
        return {}
    return dict(_deserialize(db_input_data))