from ..diffusion import dtiaverage
def test_dtiaverage_inputs():
    input_map = dict(DTI_double=dict(argstr='--DTI_double '), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputs=dict(argstr='--inputs %s...'), tensor_output=dict(argstr='--tensor_output %s', hash_files=False), verbose=dict(argstr='--verbose '))
    inputs = dtiaverage.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value