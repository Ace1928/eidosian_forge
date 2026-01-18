from ..histogrammatching import HistogramMatching
def test_HistogramMatching_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = HistogramMatching.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value