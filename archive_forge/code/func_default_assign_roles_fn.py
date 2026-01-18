import importlib
import math
import re
from enum import Enum
def default_assign_roles_fn(agents):
    """
    Assign agent role.

    Default role assignment.

    :param:
        list of agents
    """
    for i, a in enumerate(agents):
        a.disp_id = f'Agent_{i}'