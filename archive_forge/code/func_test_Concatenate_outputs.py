from ..model import Concatenate
def test_Concatenate_outputs():
    output_map = dict(concatenated_file=dict(extensions=None))
    outputs = Concatenate.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value