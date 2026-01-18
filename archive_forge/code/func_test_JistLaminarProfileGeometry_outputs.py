from ..developer import JistLaminarProfileGeometry
def test_JistLaminarProfileGeometry_outputs():
    output_map = dict(outResult=dict(extensions=None))
    outputs = JistLaminarProfileGeometry.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value