from ..convert import VtkStreamlines
def test_VtkStreamlines_outputs():
    output_map = dict(vtk=dict(extensions=None))
    outputs = VtkStreamlines.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value