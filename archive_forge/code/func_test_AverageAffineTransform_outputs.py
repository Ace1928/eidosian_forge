from ..utils import AverageAffineTransform
def test_AverageAffineTransform_outputs():
    output_map = dict(affine_transform=dict(extensions=None))
    outputs = AverageAffineTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value