from ..possum import B0Calc
def test_B0Calc_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = B0Calc.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value