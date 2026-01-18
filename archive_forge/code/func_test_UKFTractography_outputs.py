from ..ukftractography import UKFTractography
def test_UKFTractography_outputs():
    output_map = dict(tracts=dict(extensions=None), tractsWithSecondTensor=dict(extensions=None))
    outputs = UKFTractography.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value