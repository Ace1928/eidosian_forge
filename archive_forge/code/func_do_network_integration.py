from __future__ import annotations
import argparse
import collections.abc as c
import os
import typing as t
from ....commands.integration.network import (
from ....config import (
from ....target import (
from ....data import (
from ...environments import (
from ...completers import (
def do_network_integration(subparsers, parent: argparse.ArgumentParser, add_integration_common: c.Callable[[argparse.ArgumentParser], None], completer: CompositeActionCompletionFinder):
    """Command line parsing for the `network-integration` command."""
    parser: argparse.ArgumentParser = subparsers.add_parser('network-integration', parents=[parent], help='network integration tests')
    parser.set_defaults(func=command_network_integration, targets_func=walk_network_integration_targets, config=NetworkIntegrationConfig)
    network_integration = t.cast(argparse.ArgumentParser, parser.add_argument_group(title='network integration test arguments'))
    add_integration_common(network_integration)
    register_completer(network_integration.add_argument('--testcase', metavar='TESTCASE', help='limit a test to a specified testcase'), complete_network_testcase)
    add_environments(parser, completer, ControllerMode.DELEGATED, TargetMode.NETWORK_INTEGRATION)