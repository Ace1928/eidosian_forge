from ..specialized import BinaryMaskEditorBasedOnLandmarks
def test_BinaryMaskEditorBasedOnLandmarks_outputs():
    output_map = dict(outputBinaryVolume=dict(extensions=None))
    outputs = BinaryMaskEditorBasedOnLandmarks.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value