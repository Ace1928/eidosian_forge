from ..specialized import RobustStatisticsSegmenter
def test_RobustStatisticsSegmenter_outputs():
    output_map = dict(segmentedImageFileName=dict(extensions=None, position=-1))
    outputs = RobustStatisticsSegmenter.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value