from nipype.interfaces.ants import (
import os
import pytest
@pytest.fixture()
def change_dir(request):
    orig_dir = os.getcwd()
    filepath = os.path.dirname(os.path.realpath(__file__))
    datadir = os.path.realpath(os.path.join(filepath, '../../../testing/data'))
    os.chdir(datadir)

    def move2orig():
        os.chdir(orig_dir)
    request.addfinalizer(move2orig)