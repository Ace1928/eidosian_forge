from ..preprocess import MRTrixInfo
def test_MRTrixInfo_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-2))
    inputs = MRTrixInfo.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value