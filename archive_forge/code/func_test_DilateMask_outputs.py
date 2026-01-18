from ..featuredetection import DilateMask
def test_DilateMask_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = DilateMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value