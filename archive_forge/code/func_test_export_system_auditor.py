from __future__ import absolute_import, division, print_function
import pytest
from awx.main.models.execution_environments import ExecutionEnvironment
from awx.main.models.jobs import JobTemplate
from awx.main.tests.functional.conftest import user, system_auditor  # noqa: F401; pylint: disable=unused-import
@pytest.mark.django_db
def test_export_system_auditor(run_module, organization, system_auditor):
    """
    This test illustrates that export of resources can now happen
    when ran as non-root user (i.e. system auditor). The OPTIONS
    endpoint does NOT return POST for a system auditor, but now we
    make a best-effort to parse the description string, which will
    often have the fields.
    """
    result = run_module('export', dict(all=True), system_auditor)
    assert not result.get('failed', False), result.get('msg', result)
    assert 'msg' not in result
    assert 'assets' in result
    find_by(result['assets'], 'organizations', 'name', 'Default')