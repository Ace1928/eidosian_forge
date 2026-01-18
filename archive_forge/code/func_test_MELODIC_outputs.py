from ..model import MELODIC
def test_MELODIC_outputs():
    output_map = dict(out_dir=dict(), report_dir=dict())
    outputs = MELODIC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value