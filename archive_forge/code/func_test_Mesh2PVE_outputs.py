from ..utils import Mesh2PVE
def test_Mesh2PVE_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Mesh2PVE.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value