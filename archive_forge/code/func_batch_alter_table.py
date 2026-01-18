from __future__ import annotations
from contextlib import contextmanager
import re
import textwrap
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List  # noqa
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence  # noqa
from typing import Tuple
from typing import Type  # noqa
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from sqlalchemy.sql.elements import conv
from . import batch
from . import schemaobj
from .. import util
from ..util import sqla_compat
from ..util.compat import formatannotation_fwdref
from ..util.compat import inspect_formatargspec
from ..util.compat import inspect_getfullargspec
from ..util.sqla_compat import _literal_bindparam
@contextmanager
def batch_alter_table(self, table_name: str, schema: Optional[str]=None, recreate: Literal['auto', 'always', 'never']='auto', partial_reordering: Optional[Tuple[Any, ...]]=None, copy_from: Optional[Table]=None, table_args: Tuple[Any, ...]=(), table_kwargs: Mapping[str, Any]=util.immutabledict(), reflect_args: Tuple[Any, ...]=(), reflect_kwargs: Mapping[str, Any]=util.immutabledict(), naming_convention: Optional[Dict[str, str]]=None) -> Iterator[BatchOperations]:
    """Invoke a series of per-table migrations in batch.

        Batch mode allows a series of operations specific to a table
        to be syntactically grouped together, and allows for alternate
        modes of table migration, in particular the "recreate" style of
        migration required by SQLite.

        "recreate" style is as follows:

        1. A new table is created with the new specification, based on the
           migration directives within the batch, using a temporary name.

        2. the data copied from the existing table to the new table.

        3. the existing table is dropped.

        4. the new table is renamed to the existing table name.

        The directive by default will only use "recreate" style on the
        SQLite backend, and only if directives are present which require
        this form, e.g. anything other than ``add_column()``.   The batch
        operation on other backends will proceed using standard ALTER TABLE
        operations.

        The method is used as a context manager, which returns an instance
        of :class:`.BatchOperations`; this object is the same as
        :class:`.Operations` except that table names and schema names
        are omitted.  E.g.::

            with op.batch_alter_table("some_table") as batch_op:
                batch_op.add_column(Column("foo", Integer))
                batch_op.drop_column("bar")

        The operations within the context manager are invoked at once
        when the context is ended.   When run against SQLite, if the
        migrations include operations not supported by SQLite's ALTER TABLE,
        the entire table will be copied to a new one with the new
        specification, moving all data across as well.

        The copy operation by default uses reflection to retrieve the current
        structure of the table, and therefore :meth:`.batch_alter_table`
        in this mode requires that the migration is run in "online" mode.
        The ``copy_from`` parameter may be passed which refers to an existing
        :class:`.Table` object, which will bypass this reflection step.

        .. note::  The table copy operation will currently not copy
           CHECK constraints, and may not copy UNIQUE constraints that are
           unnamed, as is possible on SQLite.   See the section
           :ref:`sqlite_batch_constraints` for workarounds.

        :param table_name: name of table
        :param schema: optional schema name.
        :param recreate: under what circumstances the table should be
         recreated. At its default of ``"auto"``, the SQLite dialect will
         recreate the table if any operations other than ``add_column()``,
         ``create_index()``, or ``drop_index()`` are
         present. Other options include ``"always"`` and ``"never"``.
        :param copy_from: optional :class:`~sqlalchemy.schema.Table` object
         that will act as the structure of the table being copied.  If omitted,
         table reflection is used to retrieve the structure of the table.

         .. seealso::

            :ref:`batch_offline_mode`

            :paramref:`~.Operations.batch_alter_table.reflect_args`

            :paramref:`~.Operations.batch_alter_table.reflect_kwargs`

        :param reflect_args: a sequence of additional positional arguments that
         will be applied to the table structure being reflected / copied;
         this may be used to pass column and constraint overrides to the
         table that will be reflected, in lieu of passing the whole
         :class:`~sqlalchemy.schema.Table` using
         :paramref:`~.Operations.batch_alter_table.copy_from`.
        :param reflect_kwargs: a dictionary of additional keyword arguments
         that will be applied to the table structure being copied; this may be
         used to pass additional table and reflection options to the table that
         will be reflected, in lieu of passing the whole
         :class:`~sqlalchemy.schema.Table` using
         :paramref:`~.Operations.batch_alter_table.copy_from`.
        :param table_args: a sequence of additional positional arguments that
         will be applied to the new :class:`~sqlalchemy.schema.Table` when
         created, in addition to those copied from the source table.
         This may be used to provide additional constraints such as CHECK
         constraints that may not be reflected.
        :param table_kwargs: a dictionary of additional keyword arguments
         that will be applied to the new :class:`~sqlalchemy.schema.Table`
         when created, in addition to those copied from the source table.
         This may be used to provide for additional table options that may
         not be reflected.
        :param naming_convention: a naming convention dictionary of the form
         described at :ref:`autogen_naming_conventions` which will be applied
         to the :class:`~sqlalchemy.schema.MetaData` during the reflection
         process.  This is typically required if one wants to drop SQLite
         constraints, as these constraints will not have names when
         reflected on this backend.  Requires SQLAlchemy **0.9.4** or greater.

         .. seealso::

            :ref:`dropping_sqlite_foreign_keys`

        :param partial_reordering: a list of tuples, each suggesting a desired
         ordering of two or more columns in the newly created table.  Requires
         that :paramref:`.batch_alter_table.recreate` is set to ``"always"``.
         Examples, given a table with columns "a", "b", "c", and "d":

         Specify the order of all columns::

            with op.batch_alter_table(
                "some_table",
                recreate="always",
                partial_reordering=[("c", "d", "a", "b")],
            ) as batch_op:
                pass

         Ensure "d" appears before "c", and "b", appears before "a"::

            with op.batch_alter_table(
                "some_table",
                recreate="always",
                partial_reordering=[("d", "c"), ("b", "a")],
            ) as batch_op:
                pass

         The ordering of columns not included in the partial_reordering
         set is undefined.   Therefore it is best to specify the complete
         ordering of all columns for best results.

        .. note:: batch mode requires SQLAlchemy 0.8 or above.

        .. seealso::

            :ref:`batch_migrations`

        """
    impl = batch.BatchOperationsImpl(self, table_name, schema, recreate, copy_from, table_args, table_kwargs, reflect_args, reflect_kwargs, naming_convention, partial_reordering)
    batch_op = BatchOperations(self.migration_context, impl=impl)
    yield batch_op
    impl.flush()