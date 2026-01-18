import logging
from typing import Iterable, Optional, Set, Tuple
from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message
def prepare_distribution_metadata(self, finder: PackageFinder, build_isolation: bool, check_build_deps: bool) -> None:
    self.req.load_pyproject_toml()
    should_isolate = self.req.use_pep517 and build_isolation
    if should_isolate:
        self._prepare_build_backend(finder)
        self.req.isolated_editable_sanity_check()
        self._install_build_reqs(finder)
    should_check_deps = self.req.use_pep517 and check_build_deps
    if should_check_deps:
        pyproject_requires = self.req.pyproject_requires
        assert pyproject_requires is not None
        conflicting, missing = self.req.build_env.check_requirements(pyproject_requires)
        if conflicting:
            self._raise_conflicts('the backend dependencies', conflicting)
        if missing:
            self._raise_missing_reqs(missing)
    self.req.prepare_metadata()