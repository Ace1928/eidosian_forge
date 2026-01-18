from ..registration import CompositeTransformUtil
def test_CompositeTransformUtil_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s...', mandatory=True, position=3), num_threads=dict(nohash=True, usedefault=True), out_file=dict(argstr='%s', extensions=None, position=2), output_prefix=dict(argstr='%s', position=4, usedefault=True), process=dict(argstr='--%s', position=1, usedefault=True))
    inputs = CompositeTransformUtil.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value