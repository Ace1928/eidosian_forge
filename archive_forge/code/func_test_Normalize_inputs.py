from ..preprocess import Normalize
def test_Normalize_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), gradient=dict(argstr='-g %d'), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-2), mask=dict(argstr='-mask %s', extensions=None), out_file=dict(argstr='%s', extensions=None, hash_files=False, keep_extension=True, name_source=['in_file'], name_template='%s_norm', position=-1), segmentation=dict(argstr='-aseg %s', extensions=None), subjects_dir=dict(), transform=dict(extensions=None))
    inputs = Normalize.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value