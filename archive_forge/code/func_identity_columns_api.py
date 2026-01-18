from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
@property
def identity_columns_api(self):
    return exclusions.only_if(exclusions.BooleanPredicate(sqla_compat.has_identity))