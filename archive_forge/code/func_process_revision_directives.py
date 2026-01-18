from alembic.operations import ops
from alembic.util import Dispatcher
from alembic.util import rev_id as new_rev_id
from keystone.common.sql import upgrades
from keystone.i18n import _
def process_revision_directives(context, revision, directives):
    directives[:] = list(_assign_directives(context, directives))