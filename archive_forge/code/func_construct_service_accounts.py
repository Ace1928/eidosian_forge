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
def construct_service_accounts(args, product_type):
    """Creates the three required Google service accounts or use provided.

  Args:
    args: Command line arguments.
    product_type: events_constants.Product enum.

  Returns:
    Dict[ServiceAccountConfig, GsaEmail].
  """
    gsa_emails = {}
    for sa_config in SERVICE_ACCOUNT_CONFIGS:
        gsa_emails[sa_config] = _construct_service_account_email(sa_config, args, product_type)
    return gsa_emails