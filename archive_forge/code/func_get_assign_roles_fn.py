import importlib
import math
import re
from enum import Enum
def get_assign_roles_fn(world_module, world_name):
    """
    Get assign roles function for a world.

    :param world_module:
        module. a python module encompassing the worlds
    :param world_name:
        string. the name of the world in the module

    :return:
        the assign roles function if available, else None
    """
    return get_world_fn_attr(world_module, world_name, 'assign_roles', raise_if_missing=False)