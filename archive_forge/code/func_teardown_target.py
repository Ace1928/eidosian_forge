from __future__ import annotations
import abc
import os
import shutil
import tempfile
import typing as t
import zipfile
from ...io import (
from ...ansible_util import (
from ...config import (
from ...util import (
from ...util_common import (
from ...coverage_util import (
from ...host_configs import (
from ...data import (
from ...host_profiles import (
from ...provisioning import (
from ...connections import (
from ...inventory import (
def teardown_target(self) -> None:
    """Perform teardown for code coverage on the target."""
    if not self.target_profile:
        return
    if isinstance(self.target_profile, ControllerProfile):
        return
    profile = t.cast(SshTargetHostProfile, self.target_profile)
    platform = get_coverage_platform(profile.config)
    con = profile.get_controller_target_connections()[0]
    with tempfile.NamedTemporaryFile(prefix='ansible-test-coverage-', suffix='.tgz') as coverage_tgz:
        try:
            con.create_archive(chdir=self.common_temp_path, name=ResultType.COVERAGE.name, dst=coverage_tgz)
        except SubprocessError as ex:
            display.warning(f'Failed to download coverage results: {ex}')
        else:
            coverage_tgz.seek(0)
            with tempfile.TemporaryDirectory() as temp_dir:
                local_con = LocalConnection(self.args)
                local_con.extract_archive(chdir=temp_dir, src=coverage_tgz)
                base_dir = os.path.join(temp_dir, ResultType.COVERAGE.name)
                for filename in os.listdir(base_dir):
                    shutil.copyfile(os.path.join(base_dir, filename), os.path.join(ResultType.COVERAGE.path, update_coverage_filename(filename, platform)))
    self.run_playbook('posix_coverage_teardown.yml', self.get_playbook_variables())