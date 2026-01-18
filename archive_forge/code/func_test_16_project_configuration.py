import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_16_project_configuration():
    project = central.select('/project/pyxnat_tests')
    version = central.version()
    from pyxnat.core.errors import DatabaseError
    try:
        assert project.quarantine_code() == 0
        assert project.prearchive_code() == 4, project.prearchive_code()
    except DatabaseError:
        if version['version'] == '1.7.5.2-SNAPSHOT':
            msg = 'Version 1.7.5.2-SNAPSHOT gives trouble on some machines.                    Skipping it'
            pytest.skip(msg)
    if version['version'] != '1.7.5.2-SNAPSHOT':
        try:
            assert project.current_arc() == b'arc001'
        except DatabaseError:
            msg = 'Check if current_arc is supported in XNAT version %s.' % version['version']
            print(msg)
    assert central._user in project.users()
    assert central._user in project.owners()
    assert project.user_role(central._user) == 'owner'