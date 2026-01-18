import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_03_assessor_create():
    assert not asse_1.exists()
    asse_1.create(assessors='xnat:qcAssessmentData')
    assert asse_1.exists()