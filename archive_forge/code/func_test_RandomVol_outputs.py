from ..developer import RandomVol
def test_RandomVol_outputs():
    output_map = dict(outRand1=dict(extensions=None))
    outputs = RandomVol.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value