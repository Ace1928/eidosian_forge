from ..misc import ModifyAffine
def test_ModifyAffine_inputs():
    input_map = dict(transformation_matrix=dict(usedefault=True), volumes=dict(mandatory=True))
    inputs = ModifyAffine.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value