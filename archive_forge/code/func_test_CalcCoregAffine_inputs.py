from ..utils import CalcCoregAffine
def test_CalcCoregAffine_inputs():
    input_map = dict(invmat=dict(extensions=None), mat=dict(extensions=None), matlab_cmd=dict(), mfile=dict(usedefault=True), moving=dict(copyfile=False, extensions=None, mandatory=True), paths=dict(), target=dict(extensions=None, mandatory=True), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = CalcCoregAffine.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value