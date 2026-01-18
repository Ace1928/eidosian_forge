from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self
def execute_and_instances(orm_context: ORMExecuteState) -> Union[Result[_T], IteratorResult[_TP]]:
    active_options: Union[None, QueryContext.default_load_options, Type[QueryContext.default_load_options], BulkUDCompileState.default_update_options, Type[BulkUDCompileState.default_update_options]]
    if orm_context.is_select:
        active_options = orm_context.load_options
    elif orm_context.is_update or orm_context.is_delete:
        active_options = orm_context.update_delete_options
    else:
        active_options = None
    session = orm_context.session
    assert isinstance(session, ShardedSession)

    def iter_for_shard(shard_id: ShardIdentifier) -> Union[Result[_T], IteratorResult[_TP]]:
        bind_arguments = dict(orm_context.bind_arguments)
        bind_arguments['shard_id'] = shard_id
        orm_context.update_execution_options(identity_token=shard_id)
        return orm_context.invoke_statement(bind_arguments=bind_arguments)
    for orm_opt in orm_context._non_compile_orm_options:
        if isinstance(orm_opt, set_shard_id):
            shard_id = orm_opt.shard_id
            break
    else:
        if active_options and active_options._identity_token is not None:
            shard_id = active_options._identity_token
        elif '_sa_shard_id' in orm_context.execution_options:
            shard_id = orm_context.execution_options['_sa_shard_id']
        elif 'shard_id' in orm_context.bind_arguments:
            shard_id = orm_context.bind_arguments['shard_id']
        else:
            shard_id = None
    if shard_id is not None:
        return iter_for_shard(shard_id)
    else:
        partial = []
        for shard_id in session.execute_chooser(orm_context):
            result_ = iter_for_shard(shard_id)
            partial.append(result_)
        return partial[0].merge(*partial[1:])