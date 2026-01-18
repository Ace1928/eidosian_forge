from ..brains import insertMidACPCpoint
def test_insertMidACPCpoint_outputs():
    output_map = dict(outputLandmarkFile=dict(extensions=None))
    outputs = insertMidACPCpoint.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value