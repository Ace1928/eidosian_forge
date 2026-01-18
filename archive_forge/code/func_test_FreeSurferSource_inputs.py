from ..io import FreeSurferSource
def test_FreeSurferSource_inputs():
    input_map = dict(hemi=dict(usedefault=True), subject_id=dict(mandatory=True), subjects_dir=dict(mandatory=True))
    inputs = FreeSurferSource.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value