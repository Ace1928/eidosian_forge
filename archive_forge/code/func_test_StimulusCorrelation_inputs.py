from ..rapidart import StimulusCorrelation
def test_StimulusCorrelation_inputs():
    input_map = dict(concatenated_design=dict(mandatory=True), intensity_values=dict(mandatory=True), realignment_parameters=dict(mandatory=True), spm_mat_file=dict(extensions=None, mandatory=True))
    inputs = StimulusCorrelation.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value