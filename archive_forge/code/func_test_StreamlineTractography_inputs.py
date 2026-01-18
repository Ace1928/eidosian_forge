from ..tracks import StreamlineTractography
def test_StreamlineTractography_inputs():
    input_map = dict(gfa_thresh=dict(mandatory=True, usedefault=True), in_file=dict(extensions=None, mandatory=True), in_model=dict(extensions=None), in_peaks=dict(extensions=None), min_angle=dict(mandatory=True, usedefault=True), multiprocess=dict(mandatory=True, usedefault=True), num_seeds=dict(mandatory=True, usedefault=True), out_prefix=dict(), peak_threshold=dict(mandatory=True, usedefault=True), save_seeds=dict(mandatory=True, usedefault=True), seed_coord=dict(extensions=None), seed_mask=dict(extensions=None), tracking_mask=dict(extensions=None))
    inputs = StreamlineTractography.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value