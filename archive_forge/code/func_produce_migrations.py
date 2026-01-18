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
def produce_migrations(context: MigrationContext, metadata: MetaData) -> MigrationScript:
    """Produce a :class:`.MigrationScript` structure based on schema
    comparison.

    This function does essentially what :func:`.compare_metadata` does,
    but then runs the resulting list of diffs to produce the full
    :class:`.MigrationScript` object.   For an example of what this looks like,
    see the example in :ref:`customizing_revision`.

    .. seealso::

        :func:`.compare_metadata` - returns more fundamental "diff"
        data from comparing a schema.

    """
    autogen_context = AutogenContext(context, metadata=metadata)
    migration_script = ops.MigrationScript(rev_id=None, upgrade_ops=ops.UpgradeOps([]), downgrade_ops=ops.DowngradeOps([]))
    compare._populate_migration_script(autogen_context, migration_script)
    return migration_script