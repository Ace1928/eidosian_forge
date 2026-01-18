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
def load_on_pk_identity(session: Session, statement: Union[Select, FromStatement], primary_key_identity: Optional[Tuple[Any, ...]], *, load_options: Optional[Sequence[ORMOption]]=None, refresh_state: Optional[InstanceState[Any]]=None, with_for_update: Optional[ForUpdateArg]=None, only_load_props: Optional[Iterable[str]]=None, identity_token: Optional[Any]=None, no_autoflush: bool=False, bind_arguments: Mapping[str, Any]=util.EMPTY_DICT, execution_options: _ExecuteOptions=util.EMPTY_DICT, require_pk_cols: bool=False, is_user_refresh: bool=False):
    """Load the given primary key identity from the database."""
    query = statement
    q = query._clone()
    assert not q._is_lambda_element
    if load_options is None:
        load_options = QueryContext.default_load_options
    if statement._compile_options is SelectState.default_select_compile_options:
        compile_options = ORMCompileState.default_compile_options
    else:
        compile_options = statement._compile_options
    if primary_key_identity is not None:
        mapper = query._propagate_attrs['plugin_subject']
        _get_clause, _get_params = mapper._get_clause
        if None in primary_key_identity:
            nones = {_get_params[col].key for col, value in zip(mapper.primary_key, primary_key_identity) if value is None}
            _get_clause = sql_util.adapt_criterion_to_null(_get_clause, nones)
            if len(nones) == len(primary_key_identity):
                util.warn('fully NULL primary key identity cannot load any object.  This condition may raise an error in a future release.')
        q._where_criteria = (sql_util._deep_annotate(_get_clause, {'_orm_adapt': True}),)
        params = {_get_params[primary_key].key: id_val for id_val, primary_key in zip(primary_key_identity, mapper.primary_key)}
    else:
        params = None
    if with_for_update is not None:
        version_check = True
        q._for_update_arg = with_for_update
    elif query._for_update_arg is not None:
        version_check = True
        q._for_update_arg = query._for_update_arg
    else:
        version_check = False
    if require_pk_cols and only_load_props:
        if not refresh_state:
            raise sa_exc.ArgumentError('refresh_state is required when require_pk_cols is present')
        refresh_state_prokeys = refresh_state.mapper._primary_key_propkeys
        has_changes = {key for key in refresh_state_prokeys.difference(only_load_props) if refresh_state.attrs[key].history.has_changes()}
        if has_changes:
            raise sa_exc.InvalidRequestError(f'Please flush pending primary key changes on attributes {has_changes} for mapper {refresh_state.mapper} before proceeding with a refresh')
        mp = refresh_state.mapper._props
        for p in only_load_props:
            if mp[p]._is_relationship:
                only_load_props = refresh_state_prokeys.union(only_load_props)
                break
    if refresh_state and refresh_state.load_options:
        compile_options += {'_current_path': refresh_state.load_path.parent}
        q = q.options(*refresh_state.load_options)
    new_compile_options, load_options = _set_get_options(compile_options, load_options, version_check=version_check, only_load_props=only_load_props, refresh_state=refresh_state, identity_token=identity_token, is_user_refresh=is_user_refresh)
    q._compile_options = new_compile_options
    q._order_by = None
    if no_autoflush:
        load_options += {'_autoflush': False}
    execution_options = util.EMPTY_DICT.merge_with(execution_options, {'_sa_orm_load_options': load_options})
    result = session.execute(q, params=params, execution_options=execution_options, bind_arguments=bind_arguments).unique().scalars()
    try:
        return result.one()
    except orm_exc.NoResultFound:
        return None