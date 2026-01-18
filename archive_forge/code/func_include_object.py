from logging.config import fileConfig
from alembic import context
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from keystone.common.sql import core
from keystone.common.sql.migrations import autogen
def include_object(object, name, type_, reflected, compare_to):
    BORKED_COLUMNS = ()
    BORKED_UNIQUE_CONSTRAINTS = ()
    BORKED_FK_CONSTRAINTS = (('application_credential_access_rule', ['access_rule_id']), ('limit', ['registered_limit_id']), ('registered_limit', ['service_id']), ('registered_limit', ['region_id']), ('endpoint', ['region_id']), ('application_credential_access_rule', ['access_rule_id']), ('endpoint', ['region_id']), ('assignment', ['role_id']))
    BORKED_INDEXES = (('access_rule', ['external_id']), ('access_rule', ['user_id']), ('revocation_event', ['revoked_at']), ('system_assignment', ['actor_id']), ('user', ['default_project_id']), ('access_rule', ['external_id']), ('access_rule', ['user_id']), ('access_token', ['consumer_id']), ('endpoint', ['service_id']), ('revocation_event', ['revoked_at']), ('user', ['default_project_id']), ('user_group_membership', ['group_id']), ('trust', ['trustor_user_id', 'trustee_user_id', 'project_id', 'impersonation', 'expires_at', 'expires_at_int']))
    if type_ == 'column':
        return (object.table.name, name) not in BORKED_COLUMNS
    if type_ == 'unique_constraint':
        columns = [c.name for c in object.columns]
        return (object.table.name, columns) not in BORKED_UNIQUE_CONSTRAINTS
    if type_ == 'foreign_key_constraint':
        columns = [c.name for c in object.columns]
        return (object.table.name, columns) not in BORKED_FK_CONSTRAINTS
    if type_ == 'index':
        columns = [c.name for c in object.columns]
        return (object.table.name, columns) not in BORKED_INDEXES
    return True