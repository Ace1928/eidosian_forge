from ..utils import ReHo
def test_ReHo_outputs():
    output_map = dict(out_file=dict(extensions=None), out_vals=dict(extensions=None))
    outputs = ReHo.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value