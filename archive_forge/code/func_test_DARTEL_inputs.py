from ..preprocess import DARTEL
def test_DARTEL_inputs():
    input_map = dict(image_files=dict(copyfile=False, field='warp.images', mandatory=True), iteration_parameters=dict(field='warp.settings.param'), matlab_cmd=dict(), mfile=dict(usedefault=True), optimization_parameters=dict(field='warp.settings.optim'), paths=dict(), regularization_form=dict(field='warp.settings.rform'), template_prefix=dict(field='warp.settings.template', usedefault=True), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True))
    inputs = DARTEL.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value