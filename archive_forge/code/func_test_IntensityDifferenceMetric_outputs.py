from ..changequantification import IntensityDifferenceMetric
def test_IntensityDifferenceMetric_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1), reportFileName=dict(extensions=None))
    outputs = IntensityDifferenceMetric.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value