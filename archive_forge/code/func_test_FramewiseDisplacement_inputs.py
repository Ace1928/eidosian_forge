from ..confounds import FramewiseDisplacement
def test_FramewiseDisplacement_inputs():
    input_map = dict(figdpi=dict(usedefault=True), figsize=dict(usedefault=True), in_file=dict(extensions=None, mandatory=True), normalize=dict(usedefault=True), out_figure=dict(extensions=None, usedefault=True), out_file=dict(extensions=None, usedefault=True), parameter_source=dict(mandatory=True), radius=dict(usedefault=True), save_plot=dict(usedefault=True), series_tr=dict())
    inputs = FramewiseDisplacement.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value