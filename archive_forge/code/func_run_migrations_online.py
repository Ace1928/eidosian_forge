from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from keystone.common.sql import core
from keystone.common.sql.migrations import autogen
def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a connection
    with the context.

    This is modified from the default based on the below, since we want to
    share an engine when unit testing so in-memory database testing actually
    works.

    https://alembic.sqlalchemy.org/en/latest/cookbook.html#connection-sharing
    """
    connectable = config.attributes.get('connection', None)
    if connectable is None:
        connectable = engine_from_config(config.get_section(config.config_ini_section), prefix='sqlalchemy.', poolclass=pool.NullPool)
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True, include_name=include_name, include_object=include_object, process_revision_directives=autogen.process_revision_directives)
            with context.begin_transaction():
                context.run_migrations()
    else:
        context.configure(connection=connectable, target_metadata=target_metadata, render_as_batch=True, include_name=include_name, include_object=include_object, process_revision_directives=autogen.process_revision_directives)
        with context.begin_transaction():
            context.run_migrations()