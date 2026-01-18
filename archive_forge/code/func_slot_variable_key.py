from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def slot_variable_key(variable_path, optimizer_path, slot_name):
    """Returns checkpoint key for a slot variable."""
    return f'{variable_path}/{_OPTIMIZER_SLOTS_NAME}/{optimizer_path}/{escape_local_name(slot_name)}'