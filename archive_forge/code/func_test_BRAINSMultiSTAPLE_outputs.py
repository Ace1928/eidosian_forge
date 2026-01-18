from ..specialized import BRAINSMultiSTAPLE
def test_BRAINSMultiSTAPLE_outputs():
    output_map = dict(outputConfusionMatrix=dict(extensions=None), outputMultiSTAPLE=dict(extensions=None))
    outputs = BRAINSMultiSTAPLE.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value