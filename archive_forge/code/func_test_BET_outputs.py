from ..preprocess import BET
def test_BET_outputs():
    output_map = dict(inskull_mask_file=dict(extensions=None), inskull_mesh_file=dict(extensions=None), mask_file=dict(extensions=None), meshfile=dict(extensions=None), out_file=dict(extensions=None), outline_file=dict(extensions=None), outskin_mask_file=dict(extensions=None), outskin_mesh_file=dict(extensions=None), outskull_mask_file=dict(extensions=None), outskull_mesh_file=dict(extensions=None), skull_file=dict(extensions=None), skull_mask_file=dict(extensions=None))
    outputs = BET.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value