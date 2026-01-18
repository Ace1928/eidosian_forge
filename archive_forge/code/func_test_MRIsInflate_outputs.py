from ..utils import MRIsInflate
def test_MRIsInflate_outputs():
    output_map = dict(out_file=dict(extensions=None), out_sulc=dict(extensions=None))
    outputs = MRIsInflate.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value