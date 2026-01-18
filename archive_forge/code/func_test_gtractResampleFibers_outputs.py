from ..gtract import gtractResampleFibers
def test_gtractResampleFibers_outputs():
    output_map = dict(outputTract=dict(extensions=None))
    outputs = gtractResampleFibers.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value