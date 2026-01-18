from ..specialized import VBRAINSDemonWarp
def test_VBRAINSDemonWarp_outputs():
    output_map = dict(outputCheckerboardVolume=dict(extensions=None), outputDisplacementFieldVolume=dict(extensions=None), outputVolume=dict(extensions=None))
    outputs = VBRAINSDemonWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value