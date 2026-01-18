from ..dti import DTLUTGen
def test_DTLUTGen_outputs():
    output_map = dict(dtLUT=dict(extensions=None))
    outputs = DTLUTGen.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value