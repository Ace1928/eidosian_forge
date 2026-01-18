from ..utils import TalairachAVI
def test_TalairachAVI_outputs():
    output_map = dict(out_file=dict(extensions=None), out_log=dict(extensions=None), out_txt=dict(extensions=None))
    outputs = TalairachAVI.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value