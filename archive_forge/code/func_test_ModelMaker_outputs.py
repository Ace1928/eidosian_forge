from ..surface import ModelMaker
def test_ModelMaker_outputs():
    output_map = dict(modelSceneFile=dict())
    outputs = ModelMaker.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value