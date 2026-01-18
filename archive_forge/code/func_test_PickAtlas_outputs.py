from ..misc import PickAtlas
def test_PickAtlas_outputs():
    output_map = dict(mask_file=dict(extensions=None))
    outputs = PickAtlas.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value