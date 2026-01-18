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
def per_property_dependencies(self, uow, parent_saves, child_saves, parent_deletes, child_deletes, after_save, before_delete):
    uow.dependencies.update([(parent_saves, after_save), (child_saves, after_save), (after_save, child_deletes), (before_delete, parent_saves), (before_delete, parent_deletes), (before_delete, child_deletes), (before_delete, child_saves)])