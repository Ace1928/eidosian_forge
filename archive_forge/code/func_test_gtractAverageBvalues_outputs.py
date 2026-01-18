from ..gtract import gtractAverageBvalues
def test_gtractAverageBvalues_outputs():
    output_map = dict(outputVolume=dict(extensions=None))
    outputs = gtractAverageBvalues.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value