from ..utils import Reslice
def test_Reslice_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True), interp=dict(usedefault=True), matlab_cmd=dict(), mfile=dict(usedefault=True), out_file=dict(extensions=None), paths=dict(), space_defining=dict(extensions=None, mandatory=True), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = Reslice.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value