from ..gtract import gtractResampleDWIInPlace
def test_gtractResampleDWIInPlace_outputs():
    output_map = dict(outputResampledB0=dict(extensions=None), outputVolume=dict(extensions=None))
    outputs = gtractResampleDWIInPlace.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value