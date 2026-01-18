from ..utils import TensorMetrics
def test_TensorMetrics_inputs():
    input_map = dict(args=dict(argstr='%s'), component=dict(argstr='-num %s', sep=',', usedefault=True), environ=dict(nohash=True, usedefault=True), in_file=dict(argstr='%s', extensions=None, mandatory=True, position=-1), in_mask=dict(argstr='-mask %s', extensions=None), modulate=dict(argstr='-modulate %s'), out_ad=dict(argstr='-ad %s', extensions=None), out_adc=dict(argstr='-adc %s', extensions=None), out_cl=dict(argstr='-cl %s', extensions=None), out_cp=dict(argstr='-cp %s', extensions=None), out_cs=dict(argstr='-cs %s', extensions=None), out_eval=dict(argstr='-value %s', extensions=None), out_evec=dict(argstr='-vector %s', extensions=None), out_fa=dict(argstr='-fa %s', extensions=None), out_rd=dict(argstr='-rd %s', extensions=None))
    inputs = TensorMetrics.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value