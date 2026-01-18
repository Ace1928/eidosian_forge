from ..vista import Vnifti2Image
def test_Vnifti2Image_inputs():
    input_map = dict(args=dict(argstr='%s'), attributes=dict(argstr='-attr %s', extensions=None, position=2), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='-in %s', extensions=None, mandatory=True, position=1), out_file=dict(argstr='-out %s', extensions=None, hash_files=False, keep_extension=False, name_source=['in_file'], name_template='%s.v', position=-1))
    inputs = Vnifti2Image.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value