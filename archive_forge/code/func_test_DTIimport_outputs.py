from ..diffusion import DTIimport
def test_DTIimport_outputs():
    output_map = dict(outputTensor=dict(extensions=None, position=-1))
    outputs = DTIimport.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value