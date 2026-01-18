from ..preprocess import Volreg
def test_Volreg_outputs():
    output_map = dict(md1d_file=dict(extensions=None), oned_file=dict(extensions=None), oned_matrix_save=dict(extensions=None), out_file=dict(extensions=None))
    outputs = Volreg.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value