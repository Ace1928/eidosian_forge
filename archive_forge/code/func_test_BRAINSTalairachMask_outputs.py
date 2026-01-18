from ..segmentation import BRAINSTalairachMask
def test_BRAINSTalairachMask_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = BRAINSTalairachMask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value