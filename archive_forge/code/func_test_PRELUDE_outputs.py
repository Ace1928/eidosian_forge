from ..preprocess import PRELUDE
def test_PRELUDE_outputs():
    output_map = dict(unwrapped_phase_file=dict(extensions=None))
    outputs = PRELUDE.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value