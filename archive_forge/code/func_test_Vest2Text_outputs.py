from ..utils import Vest2Text
def test_Vest2Text_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = Vest2Text.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value