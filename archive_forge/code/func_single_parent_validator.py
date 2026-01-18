from __future__ import annotations
import collections
import itertools
from typing import Any
from typing import Dict
from typing import Tuple
from typing import TYPE_CHECKING
from . import attributes
from . import exc as orm_exc
from . import interfaces
from . import loading
from . import path_registry
from . import properties
from . import query
from . import relationships
from . import unitofwork
from . import util as orm_util
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import ATTR_WAS_SET
from .base import LoaderCallableStatus
from .base import PASSIVE_OFF
from .base import PassiveFlag
from .context import _column_descriptions
from .context import ORMCompileState
from .context import ORMSelectCompileState
from .context import QueryContext
from .interfaces import LoaderStrategy
from .interfaces import StrategizedProperty
from .session import _state_session
from .state import InstanceState
from .strategy_options import Load
from .util import _none_set
from .util import AliasedClass
from .. import event
from .. import exc as sa_exc
from .. import inspect
from .. import log
from .. import sql
from .. import util
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
def single_parent_validator(desc, prop):

    def _do_check(state, value, oldvalue, initiator):
        if value is not None and initiator.key == prop.key:
            hasparent = initiator.hasparent(attributes.instance_state(value))
            if hasparent and oldvalue is not value:
                raise sa_exc.InvalidRequestError('Instance %s is already associated with an instance of %s via its %s attribute, and is only allowed a single parent.' % (orm_util.instance_str(value), state.class_, prop), code='bbf1')
        return value

    def append(state, value, initiator):
        return _do_check(state, value, None, initiator)

    def set_(state, value, oldvalue, initiator):
        return _do_check(state, value, oldvalue, initiator)
    event.listen(desc, 'append', append, raw=True, retval=True, active_history=True)
    event.listen(desc, 'set', set_, raw=True, retval=True, active_history=True)