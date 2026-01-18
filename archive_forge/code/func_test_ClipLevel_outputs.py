from ..preprocess import ClipLevel
def test_ClipLevel_outputs():
    output_map = dict(clip_val=dict())
    outputs = ClipLevel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value