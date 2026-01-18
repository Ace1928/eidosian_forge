from ..mesh import MeshWarpMaths
def test_MeshWarpMaths_outputs():
    output_map = dict(out_file=dict(extensions=None), out_warp=dict(extensions=None))
    outputs = MeshWarpMaths.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value