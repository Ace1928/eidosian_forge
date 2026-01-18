from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
def per_state_dependencies(self, uow, save_parent, delete_parent, child_action, after_save, before_delete, isdelete, childisdelete):
    if not isdelete:
        if childisdelete:
            uow.dependencies.update([(save_parent, after_save), (after_save, child_action)])
        else:
            uow.dependencies.update([(save_parent, after_save), (child_action, after_save)])
    else:
        uow.dependencies.update([(before_delete, child_action), (before_delete, delete_parent)])