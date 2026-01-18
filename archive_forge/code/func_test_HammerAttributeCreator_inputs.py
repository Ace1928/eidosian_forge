from ..featuredetection import HammerAttributeCreator
def test_HammerAttributeCreator_inputs():
    input_map = dict(Scale=dict(argstr='--Scale %d'), Strength=dict(argstr='--Strength %f'), args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), inputCSFVolume=dict(argstr='--inputCSFVolume %s', extensions=None), inputGMVolume=dict(argstr='--inputGMVolume %s', extensions=None), inputWMVolume=dict(argstr='--inputWMVolume %s', extensions=None), outputVolumeBase=dict(argstr='--outputVolumeBase %s'))
    inputs = HammerAttributeCreator.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value