from ..rapidart import StimulusCorrelation
def test_StimulusCorrelation_outputs():
    output_map = dict(stimcorr_files=dict())
    outputs = StimulusCorrelation.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value