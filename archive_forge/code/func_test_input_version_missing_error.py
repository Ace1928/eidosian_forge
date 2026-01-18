import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_input_version_missing_error(caplog):
    from nipype import config

    class DerivedInterface(nib.BaseInterface):

        class input_spec(nib.TraitedSpec):
            foo = nib.traits.Int(min_ver='0.9')
            bar = nib.traits.Int(max_ver='0.9')
        _version = 'misparsed-garbage'
    obj1 = DerivedInterface(foo=1)
    obj2 = DerivedInterface(bar=1)
    with caplog.at_level(logging.WARNING, logger='nipype.interface'):
        with mock.patch.object(config, 'getboolean', return_value=True):
            with pytest.raises(ValueError):
                obj1._check_version_requirements(obj1.inputs)
            with pytest.raises(ValueError):
                obj2._check_version_requirements(obj2.inputs)
    assert len(caplog.records) == 2