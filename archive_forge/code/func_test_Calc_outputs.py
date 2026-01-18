from ..minc import Calc
def test_Calc_outputs():
    output_map = dict(output_file=dict(extensions=None))
    outputs = Calc.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value