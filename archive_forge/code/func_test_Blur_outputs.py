from ..minc import Blur
def test_Blur_outputs():
    output_map = dict(gradient_dxyz=dict(extensions=None), output_file=dict(extensions=None), partial_dx=dict(extensions=None), partial_dxyz=dict(extensions=None), partial_dy=dict(extensions=None), partial_dz=dict(extensions=None))
    outputs = Blur.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value