from ..preprocess import NewSegment
def test_NewSegment_inputs():
    input_map = dict(affine_regularization=dict(field='warp.affreg'), channel_files=dict(copyfile=False, field='channel', mandatory=True), channel_info=dict(field='channel'), matlab_cmd=dict(), mfile=dict(usedefault=True), paths=dict(), sampling_distance=dict(field='warp.samp'), tissues=dict(field='tissue'), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True), warping_regularization=dict(field='warp.reg'), write_deformation_fields=dict(field='warp.write'))
    inputs = NewSegment.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value