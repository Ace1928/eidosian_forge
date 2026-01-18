from ..fix import AccuracyTester
def test_AccuracyTester_outputs():
    output_map = dict(output_directory=dict(argstr='%s', position=1))
    outputs = AccuracyTester.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value