from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.events import iam_util
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.iam import iam_util as core_iam_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def initialize_workload_identity_gsa(client, gsa_emails):
    """Binds GSA to KSA and allow the source GSA to assume the controller GSA."""
    for sa_config in SERVICE_ACCOUNT_CONFIGS:
        _configure_service_account_roles(sa_config, gsa_emails)
    for sa_config in [CONTROL_PLANE_SERVICE_ACCOUNT_CONFIG, BROKER_SERVICE_ACCOUNT_CONFIG]:
        _bind_eventing_gsa_to_ksa(sa_config, client, gsa_emails[sa_config].email)
    controller_sa_email = gsa_emails[CONTROL_PLANE_SERVICE_ACCOUNT_CONFIG].email
    sources_sa_email = gsa_emails[SOURCES_SERVICE_ACCOUNT_CONFIG].email
    controller_ksa = 'serviceAccount:{}'.format(controller_sa_email)
    iam_util.AddIamPolicyBindingServiceAccount(sources_sa_email, 'roles/iam.serviceAccountAdmin', controller_ksa)
    client.MarkClusterInitialized({'serviceAccountName': SOURCES_SERVICE_ACCOUNT_CONFIG.k8s_service_account, 'workloadIdentityMapping': {SOURCES_SERVICE_ACCOUNT_CONFIG.k8s_service_account: sources_sa_email}}, events_constants.Product.KUBERUN)