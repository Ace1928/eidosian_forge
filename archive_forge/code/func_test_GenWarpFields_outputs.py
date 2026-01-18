from ..legacy import GenWarpFields
def test_GenWarpFields_outputs():
    output_map = dict(affine_transformation=dict(extensions=None), input_file=dict(extensions=None), inverse_warp_field=dict(extensions=None), output_file=dict(extensions=None), warp_field=dict(extensions=None))
    outputs = GenWarpFields.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value