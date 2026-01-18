from ..gtract import gtractFiberTracking
def test_gtractFiberTracking_outputs():
    output_map = dict(outputTract=dict(extensions=None))
    outputs = gtractFiberTracking.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value