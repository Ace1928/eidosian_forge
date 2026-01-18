from __future__ import annotations
import collections.abc as c
import dataclasses
import functools
import itertools
import os
import pickle
import sys
import time
import traceback
import typing as t
from .config import (
from .util import (
from .util_common import (
from .thread import (
from .host_profiles import (
from .pypi_proxy import (
def prepare_profiles(args: TEnvironmentConfig, targets_use_pypi: bool=False, skip_setup: bool=False, requirements: t.Optional[c.Callable[[HostProfile], None]]=None) -> HostState:
    """
    Create new profiles, or load existing ones, and return them.
    If a requirements callback was provided, it will be used before configuring hosts if delegation has already been performed.
    """
    if args.host_path:
        host_state = HostState.deserialize(args, os.path.join(args.host_path, 'state.dat'))
    else:
        run_pypi_proxy(args, targets_use_pypi)
        host_state = HostState(controller_profile=t.cast(ControllerHostProfile, create_host_profile(args, args.controller, True)), target_profiles=[create_host_profile(args, target, False) for target in args.targets])
        if args.prime_containers:
            for host_profile in host_state.profiles:
                if isinstance(host_profile, DockerProfile):
                    host_profile.provision()
            raise PrimeContainers()
        ExitHandler.register(functools.partial(cleanup_profiles, host_state))

        def provision(profile: HostProfile) -> None:
            """Provision the given profile."""
            profile.provision()
            if not skip_setup:
                profile.setup()
        dispatch_jobs([(profile, WrappedThread(functools.partial(provision, profile))) for profile in host_state.profiles])
        host_state.controller_profile.configure()
    if not args.delegate:
        check_controller_python(args, host_state)
        if requirements:
            requirements(host_state.controller_profile)

        def configure(profile: HostProfile) -> None:
            """Configure the given profile."""
            profile.wait()
            if not skip_setup:
                profile.configure()
            if requirements:
                requirements(profile)
        dispatch_jobs([(profile, WrappedThread(functools.partial(configure, profile))) for profile in host_state.target_profiles])
    return host_state