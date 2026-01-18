from ..preprocess import DWIPreproc
def test_DWIPreproc_outputs():
    output_map = dict(out_file=dict(argstr='%s', extensions=None), out_fsl_bval=dict(argstr='%s', extensions=None, usedefault=True), out_fsl_bvec=dict(argstr='%s', extensions=None, usedefault=True), out_grad_mrtrix=dict(argstr='%s', extensions=None, usedefault=True))
    outputs = DWIPreproc.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value