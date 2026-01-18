from ..surface import GrayscaleModelMaker
def test_GrayscaleModelMaker_outputs():
    output_map = dict(OutputGeometry=dict(extensions=None, position=-1))
    outputs = GrayscaleModelMaker.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value