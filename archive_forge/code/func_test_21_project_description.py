import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_21_project_description():
    project = central.select.project('pyxnat_tests')
    desc = project.description()
    assert desc == 'pyxnat CI tests'