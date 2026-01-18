from ..imagelabelcombine import ImageLabelCombine
def test_ImageLabelCombine_outputs():
    output_map = dict(OutputLabelMap=dict(extensions=None, position=-1))
    outputs = ImageLabelCombine.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value