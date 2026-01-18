from ..developer import JistLaminarProfileCalculator
def test_JistLaminarProfileCalculator_outputs():
    output_map = dict(outResult=dict(extensions=None))
    outputs = JistLaminarProfileCalculator.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value