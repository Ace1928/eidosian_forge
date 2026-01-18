from ..diffusion import dtiestim
def test_dtiestim_outputs():
    output_map = dict(B0=dict(extensions=None), B0_mask_output=dict(extensions=None), idwi=dict(extensions=None), tensor_output=dict(extensions=None))
    outputs = dtiestim.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value