from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import database_operations
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def check_instance(instance_id):
    """Raise if the given instance doesn't exist."""
    try:
        instances.Get(instance_id)
    except apitools_exceptions.HttpNotFoundError:
        raise ValueError(textwrap.dedent("        Instance '{instance_id}' does not exist. Create it with:\n\n        $ gcloud spanner instances create {instance_id}\n        ".format(instance_id=instance_id)))