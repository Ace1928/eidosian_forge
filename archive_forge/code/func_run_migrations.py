from __future__ import annotations
from contextlib import contextmanager
from contextlib import nullcontext
import logging
import sys
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import ContextManager
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import Column
from sqlalchemy import literal_column
from sqlalchemy import MetaData
from sqlalchemy import PrimaryKeyConstraint
from sqlalchemy import String
from sqlalchemy import Table
from sqlalchemy.engine import Engine
from sqlalchemy.engine import url as sqla_url
from sqlalchemy.engine.strategies import MockEngineStrategy
from .. import ddl
from .. import util
from ..util import sqla_compat
from ..util.compat import EncodedIO
def run_migrations(self, **kw: Any) -> None:
    """Run the migration scripts established for this
        :class:`.MigrationContext`, if any.

        The commands in :mod:`alembic.command` will set up a function
        that is ultimately passed to the :class:`.MigrationContext`
        as the ``fn`` argument.  This function represents the "work"
        that will be done when :meth:`.MigrationContext.run_migrations`
        is called, typically from within the ``env.py`` script of the
        migration environment.  The "work function" then provides an iterable
        of version callables and other version information which
        in the case of the ``upgrade`` or ``downgrade`` commands are the
        list of version scripts to invoke.  Other commands yield nothing,
        in the case that a command wants to run some other operation
        against the database such as the ``current`` or ``stamp`` commands.

        :param \\**kw: keyword arguments here will be passed to each
         migration callable, that is the ``upgrade()`` or ``downgrade()``
         method within revision scripts.

        """
    self.impl.start_migrations()
    heads: Tuple[str, ...]
    if self.purge:
        if self.as_sql:
            raise util.CommandError("Can't use --purge with --sql mode")
        self._ensure_version_table(purge=True)
        heads = ()
    else:
        heads = self.get_current_heads()
        dont_mutate = self.opts.get('dont_mutate', False)
        if not self.as_sql and (not heads) and (not dont_mutate):
            self._ensure_version_table()
    head_maintainer = HeadMaintainer(self, heads)
    assert self._migrations_fn is not None
    for step in self._migrations_fn(heads, self):
        with self.begin_transaction(_per_migration=True):
            if self.as_sql and (not head_maintainer.heads):
                assert self.connection is not None
                self._version.create(self.connection)
            log.info('Running %s', step)
            if self.as_sql:
                self.impl.static_output('-- Running %s' % (step.short_log,))
            step.migration_fn(**kw)
            head_maintainer.update_to_step(step)
            for callback in self.on_version_apply_callbacks:
                callback(ctx=self, step=step.info, heads=set(head_maintainer.heads), run_args=kw)
    if self.as_sql and (not head_maintainer.heads):
        assert self.connection is not None
        self._version.drop(self.connection)