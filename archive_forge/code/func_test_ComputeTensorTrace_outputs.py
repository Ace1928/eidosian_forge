from ..dti import ComputeTensorTrace
def test_ComputeTensorTrace_outputs():
    output_map = dict(trace=dict(extensions=None))
    outputs = ComputeTensorTrace.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value