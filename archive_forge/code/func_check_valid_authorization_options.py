import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def check_valid_authorization_options(options, auth_plugin_name):
    """Validate authorization options, and provide helpful error messages."""
    if options.auth.get('project_id') and (not options.auth.get('domain_id')) and (not options.auth.get('domain_name')) and (not options.auth.get('project_name')) and (not options.auth.get('tenant_id')) and (not options.auth.get('tenant_name')):
        raise exc.CommandError(_('Missing parameter(s): Set either a project or a domain scope, but not both. Set a project scope with --os-project-name, OS_PROJECT_NAME, or auth.project_name. Alternatively, set a domain scope with --os-domain-name, OS_DOMAIN_NAME or auth.domain_name.'))