from ..brainsuite import Bfc
def test_Bfc_outputs():
    output_map = dict(correctionScheduleFile=dict(extensions=None), outputBiasField=dict(extensions=None), outputMRIVolume=dict(extensions=None), outputMaskedBiasField=dict(extensions=None))
    outputs = Bfc.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value