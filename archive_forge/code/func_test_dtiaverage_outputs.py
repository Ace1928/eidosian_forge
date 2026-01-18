from ..diffusion import dtiaverage
def test_dtiaverage_outputs():
    output_map = dict(tensor_output=dict(extensions=None))
    outputs = dtiaverage.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value