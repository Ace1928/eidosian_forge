from __future__ import annotations
import argparse
import collections.abc as c
import dataclasses
import enum
import os
import types
import typing as t
from ..constants import (
from ..util import (
from ..docker_util import (
from ..completion import (
from ..host_configs import (
from ..data import (
def handle_non_posix_targets(mode: TargetMode, options: LegacyHostOptions, targets: list[HostConfig]) -> list[HostConfig]:
    """Return a list of non-POSIX targets if the target mode is non-POSIX."""
    if mode == TargetMode.WINDOWS_INTEGRATION:
        if options.windows:
            targets = [WindowsRemoteConfig(name=f'windows/{version}', provider=options.remote_provider, arch=options.remote_arch) for version in options.windows]
        else:
            targets = [WindowsInventoryConfig(path=options.inventory)]
    elif mode == TargetMode.NETWORK_INTEGRATION:
        if options.platform:
            network_targets = [NetworkRemoteConfig(name=platform, provider=options.remote_provider, arch=options.remote_arch) for platform in options.platform]
            for platform, collection in options.platform_collection or []:
                for entry in network_targets:
                    if entry.platform == platform:
                        entry.collection = collection
            for platform, connection in options.platform_connection or []:
                for entry in network_targets:
                    if entry.platform == platform:
                        entry.connection = connection
            targets = t.cast(list[HostConfig], network_targets)
        else:
            targets = [NetworkInventoryConfig(path=options.inventory)]
    return targets