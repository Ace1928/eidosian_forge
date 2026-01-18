from ..preprocess import ParseDICOMDir
def test_ParseDICOMDir_inputs():
    input_map = dict(args=dict(argstr='%s'), dicom_dir=dict(argstr='--d %s', mandatory=True), dicom_info_file=dict(argstr='--o %s', extensions=None, usedefault=True), environ=dict(nohash=True, usedefault=True), sortbyrun=dict(argstr='--sortbyrun'), subjects_dir=dict(), summarize=dict(argstr='--summarize'))
    inputs = ParseDICOMDir.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value