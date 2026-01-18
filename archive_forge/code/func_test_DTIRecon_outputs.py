from ..dti import DTIRecon
def test_DTIRecon_outputs():
    output_map = dict(ADC=dict(extensions=None), B0=dict(extensions=None), FA=dict(extensions=None), FA_color=dict(extensions=None), L1=dict(extensions=None), L2=dict(extensions=None), L3=dict(extensions=None), V1=dict(extensions=None), V2=dict(extensions=None), V3=dict(extensions=None), exp=dict(extensions=None), tensor=dict(extensions=None))
    outputs = DTIRecon.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value