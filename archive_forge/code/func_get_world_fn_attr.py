import importlib
import math
import re
from enum import Enum
def get_world_fn_attr(world_module, world_name, fn_name, raise_if_missing=True):
    """
    Import and return the function from world.

    :param world_module:
        module. a python module encompassing the worlds
    :param world_name:
        string. the name of the world in the module
    :param fn_name:
        string. the name of the function in the world
    :param raise_if_missing:
        bool. if true, raise error if function not found

    :return:
        the function, if defined by the world.
    """
    result_fn = None
    try:
        DesiredWorld = getattr(world_module, world_name)
        result_fn = getattr(DesiredWorld, fn_name)
    except Exception as e:
        if raise_if_missing:
            print('Could not find {} for {}'.format(fn_name, world_name))
            raise e
    return result_fn