import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_20_project():
    project = central.select.project('pyxnat_tests')
    project.datatype()
    project.experiments()
    project.experiment('nose')