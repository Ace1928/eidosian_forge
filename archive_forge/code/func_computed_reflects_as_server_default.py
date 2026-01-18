from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def computed_reflects_as_server_default(self):
    return exclusions.closed()