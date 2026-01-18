from ..utils import Contrast
def test_Contrast_inputs():
    input_map = dict(annotation=dict(extensions=None, mandatory=True), args=dict(argstr='%s'), copy_inputs=dict(), cortex=dict(extensions=None, mandatory=True), environ=dict(nohash=True, usedefault=True), hemisphere=dict(argstr='--%s-only', mandatory=True), orig=dict(extensions=None, mandatory=True), rawavg=dict(extensions=None, mandatory=True), subject_id=dict(argstr='--s %s', mandatory=True, usedefault=True), subjects_dir=dict(), thickness=dict(extensions=None, mandatory=True), white=dict(extensions=None, mandatory=True))
    inputs = Contrast.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value