from ..utils import MRIsConvert
def test_MRIsConvert_outputs():
    output_map = dict(converted=dict(extensions=None))
    outputs = MRIsConvert.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value