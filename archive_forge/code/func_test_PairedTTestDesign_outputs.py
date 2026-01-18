from ..model import PairedTTestDesign
def test_PairedTTestDesign_outputs():
    output_map = dict(spm_mat_file=dict(extensions=None))
    outputs = PairedTTestDesign.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value