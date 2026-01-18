from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def check_constraints_w_enforcement(self):
    """Target database must support check constraints
        and also enforce them."""
    return exclusions.open()