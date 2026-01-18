from ..developer import JistIntensityMp2rageMasking
def test_JistIntensityMp2rageMasking_outputs():
    output_map = dict(outMasked=dict(extensions=None), outMasked2=dict(extensions=None), outSignal=dict(extensions=None), outSignal2=dict(extensions=None))
    outputs = JistIntensityMp2rageMasking.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value