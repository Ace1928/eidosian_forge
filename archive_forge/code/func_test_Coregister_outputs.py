from ..preprocess import Coregister
def test_Coregister_outputs():
    output_map = dict(coregistered_files=dict(), coregistered_source=dict())
    outputs = Coregister.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value