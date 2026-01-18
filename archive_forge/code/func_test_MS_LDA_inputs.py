from ..model import MS_LDA
def test_MS_LDA_inputs():
    input_map = dict(args=dict(argstr='%s'), conform=dict(argstr='-conform'), environ=dict(nohash=True, usedefault=True), images=dict(argstr='%s', copyfile=False, mandatory=True, position=-1), label_file=dict(argstr='-label %s', extensions=None), lda_labels=dict(argstr='-lda %s', mandatory=True, sep=' '), mask_file=dict(argstr='-mask %s', extensions=None), shift=dict(argstr='-shift %d'), subjects_dir=dict(), use_weights=dict(argstr='-W'), vol_synth_file=dict(argstr='-synth %s', extensions=None, mandatory=True), weight_file=dict(argstr='-weight %s', extensions=None, mandatory=True))
    inputs = MS_LDA.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value