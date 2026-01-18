from ..model import Remlfit
def test_Remlfit_outputs():
    output_map = dict(errts_file=dict(extensions=None), fitts_file=dict(extensions=None), glt_file=dict(extensions=None), obeta=dict(extensions=None), obuck=dict(extensions=None), oerrts=dict(extensions=None), ofitts=dict(extensions=None), oglt=dict(extensions=None), out_file=dict(extensions=None), ovar=dict(extensions=None), rbeta_file=dict(extensions=None), var_file=dict(extensions=None), wherr_file=dict(extensions=None))
    outputs = Remlfit.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value