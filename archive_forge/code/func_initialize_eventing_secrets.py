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
def initialize_eventing_secrets(client, gsa_emails, product_type):
    """Initializes eventing cluster binding three gsa's with roles and secrets.

  Args:
    client: An api_tools client.
    gsa_emails: A Dict[ServiceAccountConfig, GsaEmail] holds the gsa email and
      if the email was user provided.
    product_type: events_constants.Product enum.
  """
    for sa_config in SERVICE_ACCOUNT_CONFIGS:
        _configure_service_account_roles(sa_config, gsa_emails)
        _add_secret_to_service_account(client, sa_config, product_type, gsa_emails[sa_config].email)
        log.status.Print('Finished configuring service account for {}.\n'.format(sa_config.description))
    cluster_defaults = {'secret': {'key': 'key.json', 'name': 'google-cloud-key'}}
    client.MarkClusterInitialized(cluster_defaults, product_type)