import collections
import copy
import functools
import itertools
import math
from oslo_log import log as logging
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import timeutils
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import function
from heat.engine import output
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import support
from heat.scaling import rolling_update
from heat.scaling import template as scl_template
def replace_priority(res_item):
    name, defn = res_item
    try:
        index = names.index(name)
    except ValueError:
        return 0
    else:
        if index < targ_cap:
            return targ_cap - index
        else:
            return total_capacity