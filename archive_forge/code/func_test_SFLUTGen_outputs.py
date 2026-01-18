from ..calib import SFLUTGen
def test_SFLUTGen_outputs():
    output_map = dict(lut_one_fibre=dict(extensions=None), lut_two_fibres=dict(extensions=None))
    outputs = SFLUTGen.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value