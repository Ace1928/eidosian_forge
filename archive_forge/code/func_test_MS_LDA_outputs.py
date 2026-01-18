from ..model import MS_LDA
def test_MS_LDA_outputs():
    output_map = dict(vol_synth_file=dict(extensions=None), weight_file=dict(extensions=None))
    outputs = MS_LDA.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value