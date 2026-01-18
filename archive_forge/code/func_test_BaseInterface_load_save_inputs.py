import os
import simplejson as json
import logging
import pytest
from unittest import mock
from .... import config
from ....testing import example_data
from ... import base as nib
from ..support import _inputs_help
def test_BaseInterface_load_save_inputs(tmpdir):
    tmp_json = tmpdir.join('settings.json').strpath

    class InputSpec(nib.TraitedSpec):
        input1 = nib.traits.Int()
        input2 = nib.traits.Float()
        input3 = nib.traits.Bool()
        input4 = nib.traits.Str()

    class DerivedInterface(nib.BaseInterface):
        input_spec = InputSpec

        def __init__(self, **inputs):
            super(DerivedInterface, self).__init__(**inputs)
    inputs_dict = {'input1': 12, 'input3': True, 'input4': 'some string'}
    bif = DerivedInterface(**inputs_dict)
    bif.save_inputs_to_json(tmp_json)
    bif2 = DerivedInterface()
    bif2.load_inputs_from_json(tmp_json)
    assert bif2.inputs.get_traitsfree() == inputs_dict
    bif3 = DerivedInterface(from_file=tmp_json)
    assert bif3.inputs.get_traitsfree() == inputs_dict
    inputs_dict2 = inputs_dict.copy()
    inputs_dict2.update({'input4': 'some other string'})
    bif4 = DerivedInterface(from_file=tmp_json, input4=inputs_dict2['input4'])
    assert bif4.inputs.get_traitsfree() == inputs_dict2
    bif5 = DerivedInterface(input4=inputs_dict2['input4'])
    bif5.load_inputs_from_json(tmp_json, overwrite=False)
    assert bif5.inputs.get_traitsfree() == inputs_dict2
    bif6 = DerivedInterface(input4=inputs_dict2['input4'])
    bif6.load_inputs_from_json(tmp_json)
    assert bif6.inputs.get_traitsfree() == inputs_dict
    from nipype.interfaces.ants import Registration
    settings = example_data(example_data('smri_ants_registration_settings.json'))
    with open(settings) as setf:
        data_dict = json.load(setf)
    tsthash = Registration()
    tsthash.load_inputs_from_json(settings)
    assert {} == check_dict(data_dict, tsthash.inputs.get_traitsfree())
    tsthash2 = Registration(from_file=settings)
    assert {} == check_dict(data_dict, tsthash2.inputs.get_traitsfree())
    _, hashvalue = tsthash.inputs.get_hashval(hash_method='timestamp')
    assert hashvalue == 'e35bf07fea8049cc02de9235f85e8903'