from ..utils import ApplyTransform
def test_ApplyTransform_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = ApplyTransform.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value