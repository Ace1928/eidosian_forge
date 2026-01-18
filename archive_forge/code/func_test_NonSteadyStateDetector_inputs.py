from ..confounds import NonSteadyStateDetector
def test_NonSteadyStateDetector_inputs():
    input_map = dict(in_file=dict(extensions=None, mandatory=True))
    inputs = NonSteadyStateDetector.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value