import contextlib
import functools
import inspect
import operator
import threading
import warnings
import debtcollector.moves
import debtcollector.removals
import debtcollector.renames
from oslo_config import cfg
from oslo_utils import excutils
from oslo_db import exception
from oslo_db import options
from oslo_db.sqlalchemy import engines
from oslo_db.sqlalchemy import orm
from oslo_db import warning
@classmethod
def is_set_w_conf(cls, value, conf, key):
    if hasattr(conf.database, key):
        opt = conf.database._group._opts[key]['opt']
        group = conf.database._group
        if opt.deprecated_for_removal and conf.get_location(key, group=group.name).location == cfg.Locations.opt_default:
            return False
        return True
    return cls.is_set(value)