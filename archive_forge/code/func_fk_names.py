from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def fk_names(self):
    return exclusions.open()