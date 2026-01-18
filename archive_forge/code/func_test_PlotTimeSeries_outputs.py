from ..utils import PlotTimeSeries
def test_PlotTimeSeries_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = PlotTimeSeries.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value