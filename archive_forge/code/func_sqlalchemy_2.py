from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def sqlalchemy_2(self):
    return exclusions.skip_if(lambda config: not util.sqla_2, 'SQLAlchemy 2.x test')