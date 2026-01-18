import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_06_multi_create():
    asse_2 = central.select('/projects/pyxnat_tests/subjects/%(sid)s/experiments/%(eid)s/assessors/%(aid)s' % _id_set2)
    expe_2 = central.select('/projects/pyxnat_tests/subjects/%(sid)s/experiments/%(eid)s' % _id_set2)
    assert not asse_2.exists()
    asse_2.create(experiments='xnat:petSessionData', assessors='xnat:qcAssessmentData')
    assert asse_2.exists()
    assert asse_2.datatype() == 'xnat:qcAssessmentData'
    assert expe_2.datatype() == 'xnat:petSessionData'
    scan_2 = central.select('/projects/pyxnat_tests/subjects/%(sid)s/experiments/%(eid)s/scans/%(cid)s' % _id_set2).create()
    assert scan_2.datatype() == 'xnat:mrScanData'