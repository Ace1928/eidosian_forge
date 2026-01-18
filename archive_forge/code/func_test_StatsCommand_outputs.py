from ..stats import StatsCommand
def test_StatsCommand_outputs():
    output_map = dict(output=dict())
    outputs = StatsCommand.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value