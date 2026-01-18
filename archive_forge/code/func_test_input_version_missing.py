import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_input_version_missing(caplog):

    class DerivedInterface(nib.BaseInterface):

        class input_spec(nib.TraitedSpec):
            foo = nib.traits.Int(min_ver='0.9')
            bar = nib.traits.Int(max_ver='0.9')
        _version = 'misparsed-garbage'
    obj = DerivedInterface()
    obj.inputs.foo = 1
    obj.inputs.bar = 1
    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        obj._check_version_requirements(obj.inputs)
    assert len(caplog.records) == 2