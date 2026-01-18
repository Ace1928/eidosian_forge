from ..visualization import CreateTiledMosaic
def test_CreateTiledMosaic_outputs():
    output_map = dict(output_image=dict(extensions=None))
    outputs = CreateTiledMosaic.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value