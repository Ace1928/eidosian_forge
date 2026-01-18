from ..preprocess import Realign
def test_Realign_inputs():
    input_map = dict(fwhm=dict(field='eoptions.fwhm'), in_files=dict(copyfile=True, field='data', mandatory=True), interp=dict(field='eoptions.interp'), jobtype=dict(usedefault=True), matlab_cmd=dict(), mfile=dict(usedefault=True), out_prefix=dict(field='roptions.prefix', usedefault=True), paths=dict(), quality=dict(field='eoptions.quality'), register_to_mean=dict(field='eoptions.rtm'), separation=dict(field='eoptions.sep'), use_mcr=dict(), use_v8struct=dict(min_ver='8', usedefault=True), weight_img=dict(extensions=None, field='eoptions.weight'), wrap=dict(field='eoptions.wrap'), write_interp=dict(field='roptions.interp'), write_mask=dict(field='roptions.mask'), write_which=dict(field='roptions.which', maxlen=2, minlen=2, usedefault=True), write_wrap=dict(field='roptions.wrap'))
    inputs = Realign.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value