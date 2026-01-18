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
def complete_network_testcase(prefix: str, parsed_args: argparse.Namespace, **_) -> list[str]:
    """Return a list of test cases matching the given prefix if only one target was parsed from the command line, otherwise return an empty list."""
    testcases = []
    if len(parsed_args.include) != 1:
        return []
    target = parsed_args.include[0]
    test_dir = os.path.join(data_context().content.integration_targets_path, target, 'tests')
    connection_dirs = data_context().content.get_dirs(test_dir)
    for connection_dir in connection_dirs:
        for testcase in [os.path.basename(path) for path in data_context().content.get_files(connection_dir)]:
            if testcase.startswith(prefix):
                testcases.append(testcase.split('.', 1)[0])
    return testcases