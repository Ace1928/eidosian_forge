from __future__ import annotations
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import attributes
from . import exc as orm_exc
from . import path_registry
from .base import _DEFER_FOR_STATE
from .base import _RAISE_FOR_STATE
from .base import _SET_DEFERRED_EXPIRED
from .base import PassiveFlag
from .context import FromStatement
from .context import ORMCompileState
from .context import QueryContext
from .util import _none_set
from .util import state_str
from .. import exc as sa_exc
from .. import util
from ..engine import result_tuple
from ..engine.result import ChunkedIteratorResult
from ..engine.result import FrozenResult
from ..engine.result import SimpleResultMetaData
from ..sql import select
from ..sql import util as sql_util
from ..sql.selectable import ForUpdateArg
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import SelectState
from ..util import EMPTY_DICT
def load_on_ident(session: Session, statement: Union[Select, FromStatement], key: Optional[_IdentityKeyType], *, load_options: Optional[Sequence[ORMOption]]=None, refresh_state: Optional[InstanceState[Any]]=None, with_for_update: Optional[ForUpdateArg]=None, only_load_props: Optional[Iterable[str]]=None, no_autoflush: bool=False, bind_arguments: Mapping[str, Any]=util.EMPTY_DICT, execution_options: _ExecuteOptions=util.EMPTY_DICT, require_pk_cols: bool=False, is_user_refresh: bool=False):
    """Load the given identity key from the database."""
    if key is not None:
        ident = key[1]
        identity_token = key[2]
    else:
        ident = identity_token = None
    return load_on_pk_identity(session, statement, ident, load_options=load_options, refresh_state=refresh_state, with_for_update=with_for_update, only_load_props=only_load_props, identity_token=identity_token, no_autoflush=no_autoflush, bind_arguments=bind_arguments, execution_options=execution_options, require_pk_cols=require_pk_cols, is_user_refresh=is_user_refresh)