from ..developer import JistBrainPartialVolumeFilter
def test_JistBrainPartialVolumeFilter_outputs():
    output_map = dict(outPartial=dict(extensions=None))
    outputs = JistBrainPartialVolumeFilter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value