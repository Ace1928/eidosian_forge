import tempfile
from uuid import uuid1
import os.path as op
import os
from pyxnat import Interface
from pyxnat.tests import skip_if_no_network
import pytest
from pyxnat.core import interfaces
@skip_if_no_network
def test_26_subjects_delete():
    for sid in [_id_set1['sid'], _id_set2['sid']]:
        subj = central.select.project('pyxnat_tests').subject(sid)
        if subj.exists():
            subj.delete(delete_files=True)
        assert not subj.exists()