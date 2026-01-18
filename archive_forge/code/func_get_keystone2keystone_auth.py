import argparse
from keystoneauth1.identity.v3 import k2k
from keystoneauth1.loading import base
from osc_lib import exceptions as exc
from osc_lib.i18n import _
from osc_lib import utils
def get_keystone2keystone_auth(local_auth, service_provider, project_id=None, project_name=None, project_domain_id=None, project_domain_name=None):
    """Return Keystone 2 Keystone authentication for service provider.

    :param local_auth: authentication to use with the local Keystone
    :param service_provider: service provider id as registered in Keystone
    :param project_id: project id to scope to in the service provider
    :param project_name: project name to scope to in the service provider
    :param project_domain_id: id of domain in the service provider
    :param project_domain_name: name of domain to in the service provider
    :return: Keystone2Keystone auth object for service provider
    """
    return k2k.Keystone2Keystone(local_auth, service_provider, project_id=project_id, project_name=project_name, project_domain_id=project_domain_id, project_domain_name=project_domain_name)