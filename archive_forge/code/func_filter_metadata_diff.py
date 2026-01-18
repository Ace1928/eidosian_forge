import fixtures
from oslo_db import options as db_options
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_db.sqlalchemy import test_migrations
from oslo_log.fixture import logging_error as log_fixture
from oslo_log import log as logging
from oslotest import base
from keystone.common import sql
from keystone.common.sql import upgrades
import keystone.conf
from keystone.tests.unit import ksfixtures
import keystone.application_credential.backends.sql  # noqa: F401
import keystone.assignment.backends.sql  # noqa: F401
import keystone.assignment.role_backends.sql_model  # noqa: F401
import keystone.catalog.backends.sql  # noqa: F401
import keystone.credential.backends.sql  # noqa: F401
import keystone.endpoint_policy.backends.sql  # noqa: F401
import keystone.federation.backends.sql  # noqa: F401
import keystone.identity.backends.sql_model  # noqa: F401
import keystone.identity.mapping_backends.sql  # noqa: F401
import keystone.limit.backends.sql  # noqa: F401
import keystone.oauth1.backends.sql  # noqa: F401
import keystone.policy.backends.sql  # noqa: F401
import keystone.resource.backends.sql_model  # noqa: F401
import keystone.resource.config_backends.sql  # noqa: F401
import keystone.revoke.backends.sql  # noqa: F401
import keystone.trust.backends.sql  # noqa: F401
def filter_metadata_diff(self, diff):
    """Filter changes before assert in test_models_sync().

        :param diff: a list of differences (see `compare_metadata()` docs for
            details on format)
        :returns: a list of differences
        """
    new_diff = []
    for element in diff:
        if isinstance(element, list):
            if element[0][0] == 'modify_nullable':
                if (element[0][2], element[0][3]) in (('credential', 'encrypted_blob'), ('credential', 'key_hash'), ('federated_user', 'user_id'), ('federated_user', 'idp_id'), ('local_user', 'user_id'), ('nonlocal_user', 'user_id'), ('password', 'local_user_id')):
                    continue
            if element[0][0] == 'modify_default':
                if (element[0][2], element[0][3]) in (('password', 'created_at_int'), ('password', 'self_service'), ('project', 'is_domain'), ('service_provider', 'relay_state_prefix')):
                    continue
        else:
            if element[0] == 'add_index':
                if (element[1].table.name, [x.name for x in element[1].columns]) in (('access_rule', ['external_id']), ('access_rule', ['user_id']), ('revocation_event', ['revoked_at']), ('system_assignment', ['actor_id']), ('user', ['default_project_id'])):
                    continue
            if element[0] == 'remove_index':
                if (element[1].table.name, [x.name for x in element[1].columns]) in (('access_rule', ['external_id']), ('access_rule', ['user_id']), ('access_token', ['consumer_id']), ('endpoint', ['service_id']), ('revocation_event', ['revoked_at']), ('user', ['default_project_id']), ('user_group_membership', ['group_id']), ('trust', ['trustor_user_id', 'trustee_user_id', 'project_id', 'impersonation', 'expires_at', 'expires_at_int']), ()):
                    continue
            if element[0] == 'add_fk':
                if (element[1].table.name, element[1].column_keys) in (('application_credential_access_rule', ['access_rule_id']), ('limit', ['registered_limit_id']), ('registered_limit', ['service_id']), ('registered_limit', ['region_id']), ('endpoint', ['region_id'])):
                    continue
            if element[0] == 'remove_fk':
                if (element[1].table.name, element[1].column_keys) in (('application_credential_access_rule', ['access_rule_id']), ('endpoint', ['region_id']), ('assignment', ['role_id'])):
                    continue
        new_diff.append(element)
    return new_diff