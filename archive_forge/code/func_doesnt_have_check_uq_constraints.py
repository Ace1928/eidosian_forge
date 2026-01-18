from sqlalchemy.testing.requirements import Requirements
from alembic import util
from alembic.util import sqla_compat
from ..testing import exclusions
def doesnt_have_check_uq_constraints(config):
    from sqlalchemy import inspect
    insp = inspect(config.db)
    try:
        insp.get_unique_constraints('x')
    except NotImplementedError:
        return True
    except TypeError:
        return True
    except Exception:
        pass
    return False