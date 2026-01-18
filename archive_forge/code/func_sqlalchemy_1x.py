from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def sqlalchemy_1x(self):
    return exclusions.skip_if(lambda config: util.sqla_2, 'SQLAlchemy 1.x test')