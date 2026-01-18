from ..preprocess import ComputeMask
def test_ComputeMask_outputs():
    output_map = dict(brain_mask=dict(extensions=None))
    outputs = ComputeMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value