from ..converters import OrientScalarVolume
def test_OrientScalarVolume_outputs():
    output_map = dict(outputVolume=dict(extensions=None, position=-1))
    outputs = OrientScalarVolume.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value