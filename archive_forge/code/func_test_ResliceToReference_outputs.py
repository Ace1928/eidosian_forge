from ..utils import ResliceToReference
def test_ResliceToReference_outputs():
    output_map = dict(out_files=dict())
    outputs = ResliceToReference.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value