from ..fiberprocess import fiberprocess
def test_fiberprocess_outputs():
    output_map = dict(fiber_output=dict(extensions=None), voxelize=dict(extensions=None))
    outputs = fiberprocess.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value