from glob import glob
import os
from shutil import rmtree
from itertools import product
import pytest
import networkx as nx
from .... import config
from ....interfaces import utility as niu
from ... import engine as pe
from .test_base import EngineTestInterface
from .test_utils import UtilsTestInterface
def test_nested_workflow_doubleconnect():
    a = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='a')
    b = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='b')
    c = pe.Node(niu.IdentityInterface(fields=['a', 'b']), name='c')
    flow1 = pe.Workflow(name='test1')
    flow2 = pe.Workflow(name='test2')
    flow3 = pe.Workflow(name='test3')
    flow1.add_nodes([b])
    flow2.connect(a, 'a', flow1, 'b.a')
    with pytest.raises(Exception) as excinfo:
        flow3.connect(c, 'a', flow2, 'test1.b.a')
    assert 'Some connections were not found' in str(excinfo.value)
    flow3.connect(c, 'b', flow2, 'test1.b.b')