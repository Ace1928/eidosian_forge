from ..model import ThresholdStatistics
def test_ThresholdStatistics_inputs():
    input_map = dict(contrast_index=dict(mandatory=True), extent_threshold=dict(usedefault=True), height_threshold=dict(mandatory=True), matlab_cmd=dict(), mfile=dict(usedefault=True), paths=dict(), spm_mat_file=dict(copyfile=True, extensions=None, mandatory=True), stat_image=dict(copyfile=False, extensions=None, mandatory=True), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = ThresholdStatistics.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value