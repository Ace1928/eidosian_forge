from __future__ import annotations
import contextlib
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TYPE_CHECKING
from typing import Union
from sqlalchemy import inspect
from . import compare
from . import render
from .. import util
from ..operations import ops
from ..util import sqla_compat
def compare_metadata(context: MigrationContext, metadata: MetaData) -> Any:
    """Compare a database schema to that given in a
    :class:`~sqlalchemy.schema.MetaData` instance.

    The database connection is presented in the context
    of a :class:`.MigrationContext` object, which
    provides database connectivity as well as optional
    comparison functions to use for datatypes and
    server defaults - see the "autogenerate" arguments
    at :meth:`.EnvironmentContext.configure`
    for details on these.

    The return format is a list of "diff" directives,
    each representing individual differences::

        from alembic.migration import MigrationContext
        from alembic.autogenerate import compare_metadata
        from sqlalchemy import (
            create_engine,
            MetaData,
            Column,
            Integer,
            String,
            Table,
            text,
        )
        import pprint

        engine = create_engine("sqlite://")

        with engine.begin() as conn:
            conn.execute(
                text(
                    '''
                        create table foo (
                            id integer not null primary key,
                            old_data varchar,
                            x integer
                        )
                    '''
                )
            )
            conn.execute(text("create table bar (data varchar)"))

        metadata = MetaData()
        Table(
            "foo",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", Integer),
            Column("x", Integer, nullable=False),
        )
        Table("bat", metadata, Column("info", String))

        mc = MigrationContext.configure(engine.connect())

        diff = compare_metadata(mc, metadata)
        pprint.pprint(diff, indent=2, width=20)

    Output::

        [
            (
                "add_table",
                Table(
                    "bat",
                    MetaData(),
                    Column("info", String(), table=<bat>),
                    schema=None,
                ),
            ),
            (
                "remove_table",
                Table(
                    "bar",
                    MetaData(),
                    Column("data", VARCHAR(), table=<bar>),
                    schema=None,
                ),
            ),
            (
                "add_column",
                None,
                "foo",
                Column("data", Integer(), table=<foo>),
            ),
            [
                (
                    "modify_nullable",
                    None,
                    "foo",
                    "x",
                    {
                        "existing_comment": None,
                        "existing_server_default": False,
                        "existing_type": INTEGER(),
                    },
                    True,
                    False,
                )
            ],
            (
                "remove_column",
                None,
                "foo",
                Column("old_data", VARCHAR(), table=<foo>),
            ),
        ]

    :param context: a :class:`.MigrationContext`
     instance.
    :param metadata: a :class:`~sqlalchemy.schema.MetaData`
     instance.

    .. seealso::

        :func:`.produce_migrations` - produces a :class:`.MigrationScript`
        structure based on metadata comparison.

    """
    migration_script = produce_migrations(context, metadata)
    assert migration_script.upgrade_ops is not None
    return migration_script.upgrade_ops.as_diffs()