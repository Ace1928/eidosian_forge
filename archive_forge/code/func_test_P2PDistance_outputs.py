from ..mesh import P2PDistance
def test_P2PDistance_outputs():
    output_map = dict(distance=dict(), out_file=dict(extensions=None), out_warp=dict(extensions=None))
    outputs = P2PDistance.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value