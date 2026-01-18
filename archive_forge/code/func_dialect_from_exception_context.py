from oslo_utils import versionutils
from sqlalchemy import __version__
def dialect_from_exception_context(ctx):
    if sqla_2:
        return ctx.dialect
    else:
        return ctx.engine.dialect