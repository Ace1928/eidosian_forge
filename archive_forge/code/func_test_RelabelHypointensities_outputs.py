from ..utils import RelabelHypointensities
def test_RelabelHypointensities_outputs():
    output_map = dict(out_file=dict(argstr='%s', extensions=None))
    outputs = RelabelHypointensities.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value