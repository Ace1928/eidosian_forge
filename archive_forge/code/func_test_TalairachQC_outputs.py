from ..utils import TalairachQC
def test_TalairachQC_outputs():
    output_map = dict(log_file=dict(extensions=None, usedefault=True))
    outputs = TalairachQC.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value