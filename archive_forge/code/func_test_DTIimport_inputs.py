from ..diffusion import DTIimport
def test_DTIimport_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputFile=dict(argstr='%s', extensions=None, position=-2), outputTensor=dict(argstr='%s', hash_files=False, position=-1), testingmode=dict(argstr='--testingmode '))
    inputs = DTIimport.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value