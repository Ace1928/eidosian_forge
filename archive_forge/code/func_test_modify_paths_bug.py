import os
from copy import deepcopy
import pytest
from ... import engine as pe
from ....interfaces import base as nib
from ....interfaces import utility as niu
from .... import config
from ..utils import (
def test_modify_paths_bug(tmpdir):
    """
    There was a bug in which, if the current working directory contained a file with the name
    of an output String, the string would get transformed into a path, and generally wreak havoc.
    This attempts to replicate that condition, using an object with strings and paths in various
    trait configurations, to ensure that the guards added resolve the issue.
    Please see https://github.com/nipy/nipype/issues/2944 for more details.
    """
    tmpdir.chdir()
    spc = pe.Node(StrPathConfuser(in_str='2'), name='spc')
    open('2', 'w').close()
    outputs = spc.run().outputs
    out_str = outputs.out_str
    assert out_str == '2'
    out_path = outputs.out_path
    assert os.path.isabs(out_path)
    assert outputs.out_tuple == (out_path, out_str)
    assert outputs.out_dict_path == {out_str: out_path}
    assert outputs.out_dict_str == {out_str: out_str}
    assert outputs.out_list == [out_str] * 2