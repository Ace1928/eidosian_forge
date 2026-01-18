from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from os import path
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.container.fleet.policycontroller import protos
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.policycontroller import constants
from googlecloudsdk.command_lib.container.fleet.policycontroller import exceptions
from googlecloudsdk.command_lib.export import util
from googlecloudsdk.core.console import console_io
def fleet_default_cfg_flag():
    return base.Argument('--fleet-default-member-config', type=str, help='The path to a policy-controller.yaml configuration\n        file. If specified, this configuration will become the default Policy\n        Controller configuration for all memberships in your fleet. It can be\n        overridden with a membership-specific configuration by using the\n        the `Update` command.\n\n        To enable the Policy Controller Feature with a fleet-level default\n        membership configuration, run:\n\n          $ {command} --fleet-default-member-config=/path/to/policy-controller.yaml\n      ')