from ..base import IdentityInterface
def test_IdentityInterface_inputs():
    input_map = dict()
    inputs = IdentityInterface.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value