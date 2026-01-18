from ..reconstruction import EstimateResponseSH
def test_EstimateResponseSH_inputs():
    input_map = dict(auto=dict(xor=['recursive']), b0_thres=dict(usedefault=True), fa_thresh=dict(usedefault=True), in_bval=dict(extensions=None, mandatory=True), in_bvec=dict(extensions=None, mandatory=True), in_evals=dict(extensions=None, mandatory=True), in_file=dict(extensions=None, mandatory=True), in_mask=dict(extensions=None), out_mask=dict(extensions=None, usedefault=True), out_prefix=dict(), recursive=dict(xor=['auto']), response=dict(extensions=None, usedefault=True), roi_radius=dict(usedefault=True))
    inputs = EstimateResponseSH.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value