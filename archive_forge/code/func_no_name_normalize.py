from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def no_name_normalize(self):
    return exclusions.skip_if(lambda config: config.db.dialect.requires_name_normalize)