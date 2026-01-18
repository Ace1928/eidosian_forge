from pyxnat import Interface
from requests.exceptions import ConnectionError
import os.path as op
from functools import wraps
import pytest
@docker_available
def test_create_xml():
    x = Interface(config='.xnat.cfg')
    _modulepath = op.dirname(op.abspath(__file__))
    p = x.select.project('nosetests')
    s = p.subject('10001')
    e = s.experiment('10001_MR')
    fp = op.join(_modulepath, 'sess.xml')
    assert op.isfile(fp)
    e.create(xml=fp)
    xml = e.get().decode()
    assert 'Alomar' in xml