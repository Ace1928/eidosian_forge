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
def test_config_setting(tmpdir):
    tmpdir.chdir()
    wf = pe.Workflow('config')
    wf.base_dir = os.getcwd()
    crashdir = os.path.join(os.getcwd(), 'crashdir')
    os.mkdir(crashdir)
    wf.config = {'execution': {'crashdump_dir': crashdir}}
    n1 = pe.Node(niu.Function(function=_test_function4), name='errorfunc')
    wf.add_nodes([n1])
    try:
        wf.run()
    except RuntimeError:
        pass
    fl = glob(os.path.join(crashdir, 'crash*'))
    assert len(fl) == 1
    crashdir2 = os.path.join(os.getcwd(), 'crashdir2')
    os.mkdir(crashdir2)
    crashdir3 = os.path.join(os.getcwd(), 'crashdir3')
    os.mkdir(crashdir3)
    wf.config = {'execution': {'crashdump_dir': crashdir3}}
    n1.config = {'execution': {'crashdump_dir': crashdir2}}
    try:
        wf.run()
    except RuntimeError:
        pass
    fl = glob(os.path.join(crashdir2, 'crash*'))
    assert len(fl) == 1
    fl = glob(os.path.join(crashdir3, 'crash*'))
    assert len(fl) == 0