from ..model import FEAT
def test_FEAT_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), fsf_file=dict(argstr='%s', extensions=None, mandatory=True, position=0), output_type=dict())
    inputs = FEAT.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value