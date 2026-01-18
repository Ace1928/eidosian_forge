from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def foreign_key_match(self):
    return exclusions.open()