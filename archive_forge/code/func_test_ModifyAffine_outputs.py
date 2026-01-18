from ..misc import ModifyAffine
def test_ModifyAffine_outputs():
    output_map = dict(transformed_volumes=dict())
    outputs = ModifyAffine.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value