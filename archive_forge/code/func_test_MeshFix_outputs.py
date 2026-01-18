from ..meshfix import MeshFix
def test_MeshFix_outputs():
    output_map = dict(mesh_file=dict(extensions=None))
    outputs = MeshFix.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value