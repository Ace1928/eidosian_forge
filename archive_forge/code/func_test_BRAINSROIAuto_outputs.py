from ..specialized import BRAINSROIAuto
def test_BRAINSROIAuto_outputs():
    output_map = dict(outputClippedVolumeROI=dict(extensions=None), outputROIMaskVolume=dict(extensions=None))
    outputs = BRAINSROIAuto.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value