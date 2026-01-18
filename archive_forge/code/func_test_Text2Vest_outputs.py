from ..utils import Text2Vest
def test_Text2Vest_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Text2Vest.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value