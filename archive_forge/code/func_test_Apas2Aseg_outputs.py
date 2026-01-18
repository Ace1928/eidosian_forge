from ..utils import Apas2Aseg
def test_Apas2Aseg_outputs():
    output_map = dict(out_file=dict(argstr='%s', extensions=None))
    outputs = Apas2Aseg.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value